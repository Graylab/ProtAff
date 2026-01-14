import os
import torch
import numpy as np
import pandas as pd
import random
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Tuple
from omegaconf import DictConfig

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import LightningDataModule
from transformers import EsmTokenizer

@dataclass
class ConcatCollator:
    """
    Collator for 'Concat' architecture (Single Sample).
    Joins sequences BEFORE padding.
    Structure: [CLS] Binder [EOS] Target [EOS] [PAD] ...
    """
    tokenizer: Any
    max_length: int = 2048

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. Prepare Raw Strings
        eos = self.tokenizer.eos_token
        binder_seqs = [str(f["binder_seq"]).replace(":", eos) for f in features]
        target_seqs = [str(f["target_seq"]).replace(":", eos) for f in features]
        
        # 2. Tokenize WITHOUT padding/special tokens first
        b_encoded = self.tokenizer(binder_seqs, add_special_tokens=False)
        t_encoded = self.tokenizer(target_seqs, add_special_tokens=False)
        
        input_ids_list = []
        attention_mask_list = []
        
        cls_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.eos_token_id
        
        # 3. Concatenate and Truncate
        for b_ids, t_ids in zip(b_encoded["input_ids"], t_encoded["input_ids"]):
            allowed_len = self.max_length - 3
            current_len = len(b_ids) + len(t_ids)
            
            if current_len > allowed_len:
                excess = current_len - allowed_len
                # Priority: Keep Binder intact, truncate Target
                if len(t_ids) > excess:
                    t_ids = t_ids[:-excess]
                else:
                    remaining_excess = excess - len(t_ids)
                    t_ids = []
                    b_ids = b_ids[:-remaining_excess]
            
            full_ids = [cls_id] + b_ids + [eos_id] + t_ids + [eos_id]
            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            attention_mask_list.append(torch.ones(len(full_ids), dtype=torch.long))

        # 4. Pad the Batch
        pad_id = self.tokenizer.pad_token_id
        batch_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        batch_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        # 5. Process Labels
        # Labels are already normalized in the Dataset __getitem__
        reg_labels = [f["log_Aff"] for f in features]

        return {
            "input_ids": batch_ids,
            "attention_mask": batch_mask,
            "reg_labels": torch.tensor(reg_labels, dtype=torch.float32).unsqueeze(1),
        }

class AffinityDataset(Dataset):
    def __init__(self, 
                 base_df: pd.DataFrame, 
                 lookup_csv_path: str, 
                 weight_col: Optional[str] = None, 
                 balance_clusters: bool = False,
                 provided_stats: Optional[Tuple[float, float]] = None):
        """
        Args:
            provided_stats: (mean, std) tuple. 
                            If provided (Validation set), use these for normalization.
                            If None (Training set), calculate from data.
        """
        
        # 1. Load Lookup Table
        lookup_df = pd.read_csv(lookup_csv_path)
        lookup_df['key'] = lookup_df['type'].astype(str) + "_" + lookup_df['id'].astype(str)
        self.id2seq = dict(zip(lookup_df['key'], lookup_df['seq']))
        
        # 2. Map IDs (DataFrame should already be filtered by DataModule)
        self.base_df = base_df.copy()
        self.base_df["binder_key"] = "binder_" + self.base_df["binder_id"].astype(str)
        self.base_df["target_key"] = "target_" + self.base_df["target_id"].astype(str)
        
        # 3. Balancing Logic (Optional)
        self.weights = None
        if balance_clusters:
            if weight_col and weight_col in self.base_df.columns:
                stratify_col = weight_col
            else:
                stratify_col = "target_id"
            
            # Dampened weighting (inverse square root of frequency)
            counts = self.base_df[stratify_col].value_counts()
            freqs = self.base_df[stratify_col].map(counts)
            self.weights = (1.0 / np.sqrt(freqs)).to_numpy(dtype=np.float32)
            
        elif weight_col and weight_col in self.base_df.columns:
            self.weights = pd.to_numeric(self.base_df[weight_col], errors='coerce').fillna(1.0).to_numpy(dtype=np.float32)

        # Convert to dictionary for faster access
        self.data = self.base_df[["binder_key", "target_key", "log_Aff"]].to_dict('records')

        # -----------------------------------------------------------
        # 4. Z-Score Statistics (Leakage Prevention)
        # -----------------------------------------------------------
        if provided_stats:
            # Validation/Test set: Must use Training set statistics
            self.mean, self.std = provided_stats
            print(f"[Dataset] Using PROVIDED stats. Mean: {self.mean:.4f}, Std: {self.std:.4f}")
        else:
            # Training set: Calculate internal statistics
            all_labels = self.base_df["log_Aff"].values
            self.mean = float(np.mean(all_labels))
            self.std = float(np.std(all_labels))
            print(f"[Dataset] Calculated internal stats. Mean: {self.mean:.4f}, Std: {self.std:.4f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        
        raw_aff = float(row["log_Aff"])
        # Apply Z-Score Normalization
        norm_aff = (raw_aff - self.mean) / (self.std + 1e-8)

        return {
            "binder_seq": self.id2seq.get(row["binder_key"], ""),
            "target_seq": self.id2seq.get(row["target_key"], ""),
            "log_Aff": norm_aff,     
        }
    
    def get_weight(self, idx):
        return self.weights[idx] if self.weights is not None else 1.0

class AffinityDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)
        self.num_workers = cfg.data.num_workers if cfg.data.num_workers is not None else os.cpu_count()
        self.collate_fn = ConcatCollator(tokenizer=self.tokenizer, max_length=self.cfg.model.max_length)
        
        self.train_dataset = None
        self.val_dataset = None
        self.sampler = None

    def setup(self, stage: Optional[str] = None):
        print(f"[DataModule] Loading base data from {self.cfg.data.base_csv}")
        base_df = pd.read_csv(self.cfg.data.base_csv)
        
        # -----------------------------------------------------------
        # CRITICAL CHANGE: TARGET-BASED SPLIT (Unseen Targets)
        # -----------------------------------------------------------
        # 1. Identify Unique Targets
        all_targets = base_df["target_id"].unique()
        print(f"[DataModule] Found {len(all_targets)} unique targets.")

        # 2. Shuffle and Split Targets (NOT rows)
        # Ensures validation targets are completely unseen during training
        rng = np.random.default_rng(self.cfg.training.seed)
        rng.shuffle(all_targets)
        
        # 90% Train / 10% Validation split
        split_idx = int(len(all_targets) * 0.9)
        train_targets = set(all_targets[:split_idx])
        val_targets = set(all_targets[split_idx:])
        
        print(f"[DataModule] Split: {len(train_targets)} Train Targets, {len(val_targets)} Validation Targets (UNSEEN).")
        
        # 3. Filter DataFrames based on Target Split
        train_df = base_df[base_df["target_id"].isin(train_targets)].copy()
        val_df = base_df[base_df["target_id"].isin(val_targets)].copy()
        
        print(f"[DataModule] Rows: {len(train_df)} Train, {len(val_df)} Val")

        # -----------------------------------------------------------
        # 4. Create Datasets (Pass Train Stats to Val)
        # -----------------------------------------------------------
        # Train Dataset (Calculates its own mean/std)
        self.train_dataset = AffinityDataset(
            train_df, 
            self.cfg.data.lookup_csv, 
            weight_col=self.cfg.data.get("weight_col"), 
            balance_clusters=self.cfg.data.get("balance_clusters", False),
            provided_stats=None 
        )
        
        # Extract stats from Train to ensure consistency
        train_stats = (self.train_dataset.mean, self.train_dataset.std)
        
        # Val Dataset (Uses Train stats)
        self.val_dataset = AffinityDataset(
            val_df, 
            self.cfg.data.lookup_csv, 
            balance_clusters=False, 
            provided_stats=train_stats
        )

        # -----------------------------------------------------------
        # 5. Setup Sampler (For Train only)
        # -----------------------------------------------------------
        if self.train_dataset.weights is not None:
            print("[DataModule] Setting up WeightedRandomSampler for training...")
            self.sampler = WeightedRandomSampler(
                weights=self.train_dataset.weights, 
                num_samples=len(self.train_dataset), 
                replacement=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.cfg.training.batch_size, 
            collate_fn=self.collate_fn, 
            num_workers=self.num_workers, 
            sampler=self.sampler, 
            shuffle=(self.sampler is None), # Shuffle only if no sampler is used
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.cfg.training.batch_size, 
            collate_fn=self.collate_fn, 
            num_workers=self.num_workers, 
            shuffle=False, # Validation set should not be shuffled
            pin_memory=True
        )