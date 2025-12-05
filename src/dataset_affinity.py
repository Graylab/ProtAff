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
        binder_seqs = [f["binder_seq"].replace(":", eos) for f in features]
        target_seqs = [f["target_seq"].replace(":", eos) for f in features]
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
        lookup_df = pd.read_csv(lookup_csv_path)
        self.id2seq = dict(zip(lookup_df['type'] + "_" + lookup_df['id'].astype(str), lookup_df['seq']))
        
        base_df["binder_key"] = "binder_" + base_df["binder_id"].astype(str)
        base_df["target_key"] = "target_" + base_df["target_id"].astype(str)
        
        base_df = base_df[base_df['binder_key'].isin(self.id2seq) & base_df['target_key'].isin(self.id2seq)]
        
        self.weights = None
        if weight_col and weight_col in base_df.columns:
            if balance_clusters:
                counts = base_df[weight_col].value_counts()
                cluster_sizes = base_df[weight_col].map(counts)
                self.weights = (1.0 / (np.log10(cluster_sizes) + 1.0)).to_numpy(dtype=np.float32)
            else:
                self.weights = pd.to_numeric(base_df[weight_col], errors='coerce').fillna(1.0).to_numpy(dtype=np.float32)

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

class ProteinDataModule(LightningDataModule):
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
        base_df = pd.read_csv(self.cfg.data.base_csv)
        full_dataset = AffinityDataset(
            base_df, 
            self.cfg.data.lookup_csv, 
            weight_col=self.cfg.data.get("weight_col"), 
            balance_clusters=self.cfg.data.get("balance_clusters")
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

        if full_dataset.weights is not None:
            train_weights = torch.tensor([full_dataset.get_weight(i) for i in self.train_dataset.indices], dtype=torch.float)
            self.sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.cfg.training.batch_size, 
            collate_fn=self.collate_fn, num_workers=self.num_workers, 
            sampler=self.sampler, shuffle=(self.sampler is None), pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.cfg.training.batch_size, 
            collate_fn=self.collate_fn, num_workers=self.num_workers, pin_memory=True
        )