import os
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from omegaconf import DictConfig

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
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

        # 5. Process Labels (Regression Only)
        assert not any(np.isnan(f["log_Aff"]) for f in features), "Found unexpected NaN in data!"
        reg_labels = [f["log_Aff"] for f in features]

        return {
            "input_ids": batch_ids,
            "attention_mask": batch_mask,
            "reg_labels": torch.tensor(reg_labels, dtype=torch.float32).unsqueeze(1),
        }

class AffinityDataset(Dataset):
    def __init__(self, base_df: pd.DataFrame, lookup_csv_path: str, weight_col: Optional[str] = None, balance_clusters: bool = False):
        # 1. Load Lookup
        lookup_df = pd.read_csv(lookup_csv_path)
        lookup_df['key'] = lookup_df['type'].astype(str) + "_" + lookup_df['id'].astype(str)
        self.id2seq = dict(zip(lookup_df['key'], lookup_df['seq']))
        
        # 2. Map IDs
        self.base_df = base_df.copy()
        self.base_df["binder_key"] = "binder_" + self.base_df["binder_id"].astype(str)
        self.base_df["target_key"] = "target_" + self.base_df["target_id"].astype(str)
        
        # REMOVED: Backward Compatibility for 'is_binder'
        # We now assume we strictly want to train on regression (log_Aff)

        # -----------------------------------------------------------
        # 3. Balancing Logic
        # -----------------------------------------------------------
        self.weights = None
        if balance_clusters:
            # Prioritize 'cluster_id' if available
            if "cluster_id" in self.base_df.columns:
                stratify_col = "cluster_id"
            else:
                stratify_col = weight_col if weight_col and weight_col in self.base_df.columns else "target_id"
            
            print(f"[Dataset] Balancing training data based on: {stratify_col} (Dampened)")
            
            counts = self.base_df[stratify_col].value_counts()
            freqs = self.base_df[stratify_col].map(counts)
            
            # Dampened Weighting (Square Root)
            self.weights = (1.0 / np.sqrt(freqs)).to_numpy(dtype=np.float32)
            
        elif weight_col and weight_col in self.base_df.columns:
            self.weights = pd.to_numeric(self.base_df[weight_col], errors='coerce').fillna(1.0).to_numpy(dtype=np.float32)

        # 4. Convert Data to Dict
        # REMOVED: "is_binder" from selection
        self.data = self.base_df[["binder_key", "target_key", "log_Aff"]].to_dict('records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return {
            "binder_seq": self.id2seq.get(row["binder_key"], ""),
            "target_seq": self.id2seq.get(row["target_key"], ""),
            "log_Aff": float(row["log_Aff"]),   
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
        
        # 1. Create ONE Full Dataset
        full_dataset = AffinityDataset(
            base_df, 
            self.cfg.data.lookup_csv, 
            weight_col=self.cfg.data.get("weight_col"), 
            balance_clusters=self.cfg.data.get("balance_clusters", False)
        )
        
        # -----------------------------------------------------------
        # 2. RANDOM SPLIT
        # -----------------------------------------------------------
        total_len = len(full_dataset)
        val_len = int(total_len * 0.1) if total_len >= 10 else 1
        train_len = total_len - val_len

        print(f"[DataModule] Performing Random Split: {train_len} Train, {val_len} Val")
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_len, val_len],
            generator=torch.Generator().manual_seed(self.cfg.training.seed)
        )

        # -----------------------------------------------------------
        # 3. SETUP SAMPLER
        # -----------------------------------------------------------
        if full_dataset.weights is not None:
            print("[DataModule] Setting up WeightedRandomSampler for training...")
            
            # Since self.train_dataset is a Subset, we need to map indices back to the full dataset
            train_weights = torch.tensor(
                [full_dataset.get_weight(i) for i in self.train_dataset.indices], 
                dtype=torch.float
            )
            
            self.sampler = WeightedRandomSampler(
                weights=train_weights, 
                num_samples=len(train_weights), 
                replacement=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.cfg.training.batch_size, 
            collate_fn=self.collate_fn, 
            num_workers=self.num_workers, 
            sampler=self.sampler, 
            shuffle=(self.sampler is None), 
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.cfg.training.batch_size, 
            collate_fn=self.collate_fn, 
            num_workers=self.num_workers, 
            pin_memory=True
        )