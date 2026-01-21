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
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# 1. Collators (Concat vs Cross-Attention)
# ----------------------------------------------------------------------

@dataclass
class ConcatCollator:
    """
    Original 'Concat' Collator: [CLS] Binder [EOS] Target [EOS]
    """
    tokenizer: Any
    max_length: int = 2048

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        eos = self.tokenizer.eos_token
        binder_seqs = [str(f["binder_seq"]).replace(":", eos) for f in features]
        target_seqs = [str(f["target_seq"]).replace(":", eos) for f in features]
        
        b_encoded = self.tokenizer(binder_seqs, add_special_tokens=False)
        t_encoded = self.tokenizer(target_seqs, add_special_tokens=False)
        
        input_ids_list = []
        attention_mask_list = []
        cls_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.eos_token_id
        
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

        batch_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        reg_labels = [f["log_Aff"] for f in features]

        return {
            "input_ids": batch_ids,
            "attention_mask": batch_mask,
            "reg_labels": torch.tensor(reg_labels, dtype=torch.float32).unsqueeze(1),
        }

@dataclass
class CrossAttnCollator:
    """
    New 'Cross-Attention' Collator: Separate tensors for Binder and Target.
    Outputs: binder_ids, binder_mask, target_ids, target_mask
    """
    tokenizer: Any
    max_length: int = 2048

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # For Cross-Attention, we keep sequences separate
        binder_seqs = [str(f["binder_seq"]) for f in features]
        target_seqs = [str(f["target_seq"]) for f in features]
        
        # Split budget: each gets the max_length
        b_enc = self.tokenizer(
            binder_seqs, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        t_enc = self.tokenizer(
            target_seqs, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        reg_labels = [f["log_Aff"] for f in features]

        return {
            "binder_ids": b_enc["input_ids"],
            "binder_mask": b_enc["attention_mask"],
            "target_ids": t_enc["input_ids"],
            "target_mask": t_enc["attention_mask"],
            "reg_labels": torch.tensor(reg_labels, dtype=torch.float32).unsqueeze(1),
        }

# ----------------------------------------------------------------------
# 2. Dataset
# ----------------------------------------------------------------------

class AffinityDataset(Dataset):
    def __init__(self, 
                 base_df: pd.DataFrame, 
                 lookup_csv_path: str, 
                 weight_col: Optional[str] = None, 
                 balance_clusters: bool = False,
                 provided_stats: Optional[Tuple[float, float]] = None):
        
        lookup_df = pd.read_csv(lookup_csv_path)
        lookup_df['key'] = lookup_df['type'].astype(str) + "_" + lookup_df['id'].astype(str)
        self.id2seq = dict(zip(lookup_df['key'], lookup_df['seq']))
        
        self.base_df = base_df.copy()
        self.base_df["binder_key"] = "binder_" + self.base_df["binder_id"].astype(str)
        self.base_df["target_key"] = "target_" + self.base_df["target_id"].astype(str)
        
        self.weights = None
        if balance_clusters:
            stratify_col = weight_col if (weight_col and weight_col in self.base_df.columns) else "target_id"
            counts = self.base_df[stratify_col].value_counts()
            freqs = self.base_df[stratify_col].map(counts)
            self.weights = (1.0 / np.sqrt(freqs)).to_numpy(dtype=np.float32)
            
        elif weight_col and weight_col in self.base_df.columns:
            self.weights = pd.to_numeric(self.base_df[weight_col], errors='coerce').fillna(1.0).to_numpy(dtype=np.float32)

        self.data = self.base_df[["binder_key", "target_key", "log_Aff"]].to_dict('records')

        if provided_stats:
            self.mean, self.std = provided_stats
            print(f"[Dataset] Using PROVIDED stats. Mean: {self.mean:.4f}, Std: {self.std:.4f}")
        else:
            all_labels = self.base_df["log_Aff"].values
            self.mean = float(np.mean(all_labels))
            self.std = float(np.std(all_labels))
            print(f"[Dataset] Calculated internal stats. Mean: {self.mean:.4f}, Std: {self.std:.4f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        raw_aff = float(row["log_Aff"])
        norm_aff = (raw_aff - self.mean) / (self.std + 1e-8)

        return {
            "binder_seq": self.id2seq.get(row["binder_key"], ""),
            "target_seq": self.id2seq.get(row["target_key"], ""),
            "log_Aff": norm_aff,     
        }

# ----------------------------------------------------------------------
# 3. DataModule
# ----------------------------------------------------------------------
class AffinityDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)
        self.num_workers = cfg.data.num_workers if cfg.data.num_workers is not None else os.cpu_count()
        
        arch = self.cfg.model.get("arch", "concat")
        if arch in ["cross_attn", "interaction_map"]:
            print("[DataModule] Initializing CrossAttnCollator")
            self.collate_fn = CrossAttnCollator(tokenizer=self.tokenizer, max_length=self.cfg.model.max_length)
        else:
            print("[DataModule] Initializing ConcatCollator")
            self.collate_fn = ConcatCollator(tokenizer=self.tokenizer, max_length=self.cfg.model.max_length)
        
        self.weight_col = self.cfg.data.get("weight_col")
        self.split_col = self.weight_col if self.weight_col is not None else "target_id"
        
        self.train_dataset = None
        self.val_dataset = None
        self.sampler = None

    def setup(self, stage: Optional[str] = None):
        print(f"[DataModule] Split/Balance Column: {self.split_col}")
        base_df = pd.read_csv(self.cfg.data.base_csv)
        
        if self.split_col not in base_df.columns:
            raise KeyError(f"Split column '{self.split_col}' not found in CSV.")

        train_ratio = self.cfg.training.get("train_val_split", 0.9)
        strategy = self.cfg.training.get("split_strategy", "random")
        seed = self.cfg.training.seed

        if strategy == "random":
            print(f"--- Running RANDOM Split ({train_ratio*100}% Train) ---")
            train_df, val_df = train_test_split(
                base_df,
                train_size=train_ratio,
                random_state=seed,
                shuffle=True
            )
        
        elif strategy == "group":
            print(f"--- Running CUSTOM GROUP Split (Singletons to Train, Multi 9:1) ---")
            
            # 1. Count occurrences per target
            counts = base_df[self.split_col].value_counts()
            singletons = counts[counts == 1].index.tolist()
            multi_sample = counts[counts > 1].index.tolist()
            
            # 2. Shuffle multi-sample groups
            rng = np.random.default_rng(seed)
            rng.shuffle(multi_sample)
            
            # 3. Split multi-sample groups 90/10
            split_idx = int(len(multi_sample) * train_ratio)
            # Guarantee at least one group for val if possible
            if split_idx == len(multi_sample) and len(multi_sample) > 0:
                split_idx -= 1
                
            train_groups = set(multi_sample[:split_idx]).union(set(singletons))
            val_groups = set(multi_sample[split_idx:])
            
            train_df = base_df[base_df[self.split_col].isin(train_groups)].copy()
            val_df = base_df[base_df[self.split_col].isin(val_groups)].copy()

        else:
            raise ValueError(f"Unknown split_strategy: {strategy}")

        print(f"Final Counts -> Train: {len(train_df)}, Val: {len(val_df)}")

        # 3. Construct Final Datasets
        self.train_dataset = AffinityDataset(
            train_df, 
            self.cfg.data.lookup_csv, 
            weight_col=self.weight_col, 
            balance_clusters=self.cfg.data.get("balance_clusters", False)
        )
        
        self.val_dataset = AffinityDataset(
            val_df, 
            self.cfg.data.lookup_csv, 
            weight_col=self.weight_col,
            provided_stats=(self.train_dataset.mean, self.train_dataset.std)
        )

        if self.train_dataset.weights is not None:
            self.sampler = WeightedRandomSampler(
                weights=self.train_dataset.weights, 
                num_samples=len(self.train_dataset), 
                replacement=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.cfg.training.batch_size, 
            collate_fn=self.collate_fn, num_workers=self.num_workers, 
            sampler=self.sampler, shuffle=(self.sampler is None), pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.cfg.training.batch_size, 
            collate_fn=self.collate_fn, num_workers=self.num_workers, 
            shuffle=False, pin_memory=True
        )