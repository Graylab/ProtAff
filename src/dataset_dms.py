import os
import torch
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from omegaconf import DictConfig

from torch.utils.data import Dataset, DataLoader, random_split
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

        wt_encoded = self.tokenizer(
            wt_seqs, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        mut_encoded = self.tokenizer(
            mut_seqs, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        return {
            "wt_ids": wt_encoded["input_ids"],
            "wt_mask": wt_encoded["attention_mask"],
            "mut_ids": mut_encoded["input_ids"],
            "mut_mask": mut_encoded["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.float32)
        }

class DMSDataset(Dataset):
    def __init__(self, mutant_csv: str, wt_csv: str):
        self.mut_df = pd.read_csv(mutant_csv)
        self.wt_df = pd.read_csv(wt_csv)
        
        self.wt_lookup = dict(zip(self.wt_df['filename'], self.wt_df['target_seq']))
        self.mut_df = self.mut_df[self.mut_df['filename'].isin(self.wt_lookup)]
        
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

class DMSDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)
        self.num_workers = cfg.data.num_workers if cfg.data.num_workers is not None else os.cpu_count()
        self.collate_fn = DMSCollator(tokenizer=self.tokenizer, max_length=self.cfg.model.max_length)
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        mut_path = self.cfg.data.mutant_csv
        wt_path = self.cfg.data.wt_csv

        full_dataset = DMSDataset(mutant_csv=mut_path, wt_csv=wt_path)
        
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

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.cfg.training.batch_size, 
            collate_fn=self.collate_fn, 
            num_workers=self.num_workers,
            shuffle=True, 
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