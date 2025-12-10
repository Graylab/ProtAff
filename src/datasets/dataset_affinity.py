import os
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from omegaconf import DictConfig

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from pytorch_lightning import LightningDataModule
from transformers import EsmTokenizer

@dataclass
class DualStreamCollator:
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        eos = self.tokenizer.eos_token
        # Ensure we convert to string to avoid errors if pandas inferred types weirdly
        binder_seqs = [str(f["binder_seq"]).replace(":", eos) for f in features]
        target_seqs = [str(f["target_seq"]).replace(":", eos) for f in features]
        labels = [f["labels"] for f in features]

        batch_binder = self.tokenizer(
            binder_seqs, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        batch_target = self.tokenizer(
            target_seqs, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        return {
            "binder_ids": batch_binder["input_ids"],
            "binder_mask": batch_binder["attention_mask"],
            "target_ids": batch_target["input_ids"],
            "target_mask": batch_target["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.float32)
        }

class AffinityDataset(Dataset):
    def __init__(self, base_df: pd.DataFrame, lookup_csv_path: str, weight_col: Optional[str] = None, balance_clusters: bool = False):
        # 1. Load Lookup
        lookup_df = pd.read_csv(lookup_csv_path)
        # Fast dictionary creation
        lookup_df['key'] = lookup_df['type'].astype(str) + "_" + lookup_df['id'].astype(str)
        self.id2seq = dict(zip(lookup_df['key'], lookup_df['seq']))
        
        # 2. Map IDs
        base_df["binder_key"] = "binder_" + base_df["binder_id"].astype(str)
        base_df["target_key"] = "target_" + base_df["target_id"].astype(str)
        
        # 3. Filter missing sequences
        # (This vector operation is faster than iterating)
        mask = base_df['binder_key'].isin(self.id2seq) & base_df['target_key'].isin(self.id2seq)
        base_df = base_df[mask].reset_index(drop=True)
        
        # 4. Balancing Logic (Target-Centric Inverse Sqrt)
        self.weights = None
        
        if balance_clusters:
            # Default to balancing by target_id if no column provided, 
            # because your plots show Target is the imbalanced entity.
            stratify_col = weight_col if weight_col and weight_col in base_df.columns else "target_id"
            print(f"[Dataset] Balancing training data based on: {stratify_col}")

            # Count frequency of every group
            counts = base_df[stratify_col].value_counts()
            
            # Map frequency back to the individual rows
            freqs = base_df[stratify_col].map(counts)
            
            # STRATEGY: Weight = 1 / sqrt(Frequency)
            # This flattens the "skyscraper" in your plot without making rare targets explode.
            self.weights = (1.0 / np.sqrt(freqs)).to_numpy(dtype=np.float32)
            
        elif weight_col and weight_col in base_df.columns:
            # Use raw numeric weights if provided (e.g. confidence scores)
            self.weights = pd.to_numeric(base_df[weight_col], errors='coerce').fillna(1.0).to_numpy(dtype=np.float32)

        # 5. Convert Data to Dict for __getitem__ speed
        self.data = base_df[["binder_key", "target_key", "log_Aff"]].to_dict('records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return {
            "binder_seq": self.id2seq[row["binder_key"]],
            "target_seq": self.id2seq[row["target_key"]],
            "labels": row["log_Aff"]
        }
    
    def get_weight(self, idx):
        return self.weights[idx] if self.weights is not None else 1.0

class AffinityDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)
        self.num_workers = cfg.data.num_workers if cfg.data.num_workers is not None else os.cpu_count()
        self.collate_fn = DualStreamCollator(tokenizer=self.tokenizer, max_length=self.cfg.model.max_length)
        self.train_dataset = None
        self.val_dataset = None
        self.sampler = None

    def setup(self, stage: Optional[str] = None):
        # Load Data
        base_df = pd.read_csv(self.cfg.data.base_csv)
        
        # Initialize Dataset
        full_dataset = AffinityDataset(
            base_df, 
            self.cfg.data.lookup_csv, 
            weight_col=self.cfg.data.get("weight_col"), 
            balance_clusters=self.cfg.data.get("balance_clusters", False)
        )
        
        # Split Logic
        total_len = len(full_dataset)
        val_len = int(total_len * 0.1) if total_len >= 10 else 1
        train_len = total_len - val_len

        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_len, val_len],
            generator=torch.Generator().manual_seed(self.cfg.training.seed)
        )

        # Setup Weighted Sampler (Only for Training)
        if full_dataset.weights is not None:
            print("[DataModule] Setting up WeightedRandomSampler for training...")
            # We must map the subset indices back to the full dataset weights
            train_weights = torch.tensor(
                [full_dataset.get_weight(i) for i in self.train_dataset.indices], 
                dtype=torch.float
            )
            
            self.sampler = WeightedRandomSampler(
                weights=train_weights, 
                num_samples=len(train_weights), 
                replacement=True # Standard for rebalancing
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.cfg.training.batch_size, 
            collate_fn=self.collate_fn, 
            num_workers=self.num_workers, 
            sampler=self.sampler, 
            shuffle=(self.sampler is None), # Shuffle only if sampler is NOT used
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
