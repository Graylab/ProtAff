import os
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from omegaconf import DictConfig

from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from pytorch_lightning import LightningDataModule
from transformers import EsmTokenizer

@dataclass
class DMSCollator:
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        wt_seqs = [f["wt_seq"] for f in features]
        mut_seqs = [f["mut_seq"] for f in features]
        labels = [f["delta_label"] for f in features]

        # --- OPTIMIZATION: Tokenize Together ---
        all_seqs = wt_seqs + mut_seqs
        
        encoded = self.tokenizer(
            all_seqs, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        wt_ids, mut_ids = encoded["input_ids"].chunk(2, dim=0)
        wt_mask, mut_mask = encoded["attention_mask"].chunk(2, dim=0)

        return {
            "wt_ids": wt_ids,
            "wt_mask": wt_mask,
            "mut_ids": mut_ids,
            "mut_mask": mut_mask,
            "labels": torch.tensor(labels, dtype=torch.float32)
        }


class DMSDataset(Dataset):
    def __init__(self, mutant_csv: str, wt_csv: str, weight_col: Optional[str] = None, balance_clusters: bool = False):
        self.mut_df = pd.read_csv(mutant_csv)
        self.wt_df = pd.read_csv(wt_csv)
        
        self.wt_lookup = dict(zip(self.wt_df['filename'], self.wt_df['target_seq']))
        
        # Filter to ensure WT exists
        self.mut_df = self.mut_df[self.mut_df['filename'].isin(self.wt_lookup)].copy()
        
        # --- WEIGHTING LOGIC ---
        self.weights = None
        
        # 1. Strategy: Balance by Cluster Frequency (Inverse Sqrt)
        if balance_clusters:
            # Default to 'filename' if no specific weight_col provided for clustering
            stratify_col = weight_col if weight_col else 'filename'
            
            if stratify_col not in self.mut_df.columns:
                raise ValueError(f"Weight column '{stratify_col}' not found in CSV.")

            print(f"[DMS] Balancing training data based on cluster: {stratify_col}")
            counts = self.mut_df[stratify_col].value_counts()
            cluster_sizes = self.mut_df[stratify_col].map(counts)
            
            # Weight = 1 / sqrt(Frequency)
            self.weights = (1.0 / np.sqrt(cluster_sizes)).to_numpy(dtype=np.float32)

        # 2. Strategy: Use Raw Numeric Weights
        elif weight_col:
            if weight_col not in self.mut_df.columns:
                raise ValueError(f"Weight column '{weight_col}' not found in CSV.")
            
            print(f"[DMS] Using raw weights from column: {weight_col}")
            self.weights = pd.to_numeric(self.mut_df[weight_col], errors='coerce').fillna(1.0).to_numpy(dtype=np.float32)

        self.mut_data = self.mut_df.to_dict('records')

    def __len__(self):
        return len(self.mut_data)

    def __getitem__(self, idx):
        row = self.mut_data[idx]
        return {
            "wt_seq": self.wt_lookup[row['filename']],
            "mut_seq": row['mutated_sequence'],
            "delta_label": row['DMS_score_normalized']
        }
    
    def get_weight(self, idx):
        return self.weights[idx] if self.weights is not None else 1.0

class DMSDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)
        self.num_workers = cfg.data.num_workers if cfg.data.num_workers is not None else os.cpu_count()
        self.collate_fn = DMSCollator(tokenizer=self.tokenizer, max_length=self.cfg.model.max_length)
        self.train_dataset = None
        self.val_dataset = None
        self.sampler = None

    def setup(self, stage: Optional[str] = None):
        mut_path = self.cfg.data.mutant_csv
        wt_path = self.cfg.data.wt_csv
        
        # Extract Configs
        weight_col = self.cfg.data.get("weight_col", None)
        do_balance = self.cfg.data.get("balance_clusters", False)

        full_dataset = DMSDataset(
            mutant_csv=mut_path, 
            wt_csv=wt_path, 
            weight_col=weight_col, 
            balance_clusters=do_balance
        )
        
        # Consistent Split Logic
        total_len = len(full_dataset)
        if total_len < 10:
            val_len = 1
        else:
            val_len = int(total_len * 0.1)
        train_len = total_len - val_len

        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_len, val_len],
            generator=torch.Generator().manual_seed(self.cfg.training.seed)
        )

        # Setup Weighted Sampler (Training Only)
        if full_dataset.weights is not None:
            print("[DMS] Setting up WeightedRandomSampler...")
            
            # Map subset indices back to original weights
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