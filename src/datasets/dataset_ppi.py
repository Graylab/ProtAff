import os
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Tuple

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import LightningDataModule
from transformers import EsmTokenizer
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig


# ----------------------------------------------------------------------
# 1. Collators
# ----------------------------------------------------------------------

@dataclass
class PPIConcatCollator:
    """Concat Collator: [CLS] seq_a [EOS] seq_b [EOS]"""
    tokenizer: Any
    max_length: int = 512

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        seqs_a = [str(f["seq_a"]) for f in features]
        seqs_b = [str(f["seq_b"]) for f in features]
        
        a_encoded = self.tokenizer(seqs_a, add_special_tokens=False)["input_ids"]
        b_encoded = self.tokenizer(seqs_b, add_special_tokens=False)["input_ids"]
        
        cls_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        
        input_ids_list, mask_list = [], []
        
        for a_ids, b_ids in zip(a_encoded, b_encoded):
            allowed = self.max_length - 3
            if len(a_ids) + len(b_ids) > allowed:
                # Truncate b first, then a
                b_ids = b_ids[:max(0, allowed - len(a_ids))]
                a_ids = a_ids[:allowed]
            
            full_ids = [cls_id] + a_ids + [eos_id] + b_ids + [eos_id]
            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            mask_list.append(torch.ones(len(full_ids), dtype=torch.long))
        
        labels = torch.tensor([f["label"] for f in features], dtype=torch.float32)
        
        return {
            "input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id),
            "attention_mask": pad_sequence(mask_list, batch_first=True, padding_value=0),
            "labels": labels,
        }


@dataclass
class PPICrossAttnCollator:
    """Cross-Attention Collator: Separate tensors for seq_a and seq_b."""
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        seqs_a = [str(f["seq_a"]) for f in features]
        seqs_b = [str(f["seq_b"]) for f in features]
        
        a_enc = self.tokenizer(
            seqs_a, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        b_enc = self.tokenizer(
            seqs_b, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        
        labels = torch.tensor([f["label"] for f in features], dtype=torch.float32)
        
        return {
            "binder_ids": a_enc["input_ids"],
            "binder_mask": a_enc["attention_mask"],
            "target_ids": b_enc["input_ids"],
            "target_mask": b_enc["attention_mask"],
            "labels": labels,
        }


# ----------------------------------------------------------------------
# 2. Dataset
# ----------------------------------------------------------------------

class PPIDataset(Dataset):
    """PPI Dataset for confidence regression."""
    
    def __init__(
        self,
        pairs_df: pd.DataFrame,
        id2seq: Dict[str, str],
        weight_col: Optional[str] = None,
        balance_clusters: bool = False,
        balance_power: float = 0.5,
        provided_stats: Optional[Tuple[float, float]] = None,
    ):
        self.id2seq = id2seq
        self.pairs_df = pairs_df.copy()
        
        # Weights for sampling
        self.weights = None
        if balance_clusters and weight_col and weight_col in self.pairs_df.columns:
            counts = self.pairs_df[weight_col].value_counts()
            
            print(f"\n[PPIDataset] Balancing by '{weight_col}':")
            print(f"  Unique groups: {len(counts)}")
            print(f"  Max/Min samples: {counts.max()}/{counts.min()}")
            
            freqs = self.pairs_df[weight_col].map(counts)
            self.weights = (1.0 / np.power(freqs, balance_power)).to_numpy(dtype=np.float32)
            
            print(f"  Weight range: {self.weights.min():.4f} - {self.weights.max():.4f}")
        
        # Stats for normalization (optional)
        if provided_stats:
            self.mean, self.std = provided_stats
        else:
            self.mean = float(self.pairs_df["confidence"].mean())
            self.std = float(self.pairs_df["confidence"].std())
        
        print(f"[PPIDataset] Loaded {len(self):,} pairs")
        print(f"[PPIDataset] Confidence stats - Mean: {self.mean:.4f}, Std: {self.std:.4f}")
    
    def __len__(self):
        return len(self.pairs_df)
    
    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        
        # Optionally normalize confidence
        # label = (row["confidence"] - self.mean) / (self.std + 1e-8)
        label = 1.0 - row["confidence"]  # Keep raw for now
        
        return {
            "seq_a": self.id2seq.get(row["protein_a"], ""),
            "seq_b": self.id2seq.get(row["protein_b"], ""),
            "label": label,
        }


# ----------------------------------------------------------------------
# 3. DataModule
# ----------------------------------------------------------------------

class PPIDataModule(LightningDataModule):
    """DataModule for PPI pretraining."""
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)
        self.num_workers = cfg.data.get("num_workers", 4)
        
        # Select collator based on architecture
        arch = cfg.model.get("arch", "concat")
        max_length = cfg.model.get("max_length", 512)
        
        if arch in ["cross_attn", "interaction_map"]:
            print("[PPIDataModule] Using CrossAttnCollator")
            self.collate_fn = PPICrossAttnCollator(tokenizer=self.tokenizer, max_length=max_length)
        else:
            print("[PPIDataModule] Using ConcatCollator")
            self.collate_fn = PPIConcatCollator(tokenizer=self.tokenizer, max_length=max_length)
        
        self.split_col = cfg.data.get("split_col", "protein_a")
        self.weight_col = cfg.data.get("weight_col", None)
        
        self.train_dataset = None
        self.val_dataset = None
        self.sampler = None
    
    def setup(self, stage: Optional[str] = None):
        # Load data
        pairs_df = pd.read_csv(self.cfg.data.pairs_csv)
        lookup_df = pd.read_csv(self.cfg.data.lookup_csv)
        
        id2seq = dict(zip(lookup_df["id"], lookup_df["seq"]))
        
        print(f"[PPIDataModule] Loaded {len(pairs_df):,} pairs, {len(id2seq):,} proteins")
        
        # Filter by source if specified
        sources = self.cfg.data.get("sources", None)
        if sources:
            pairs_df = pairs_df[pairs_df["source"].apply(
                lambda x: any(s in str(x) for s in sources)
            )]
            print(f"[PPIDataModule] After source filter ({sources}): {len(pairs_df):,}")
        
        # Filter by confidence if specified
        min_conf = self.cfg.data.get("min_confidence", None)
        max_conf = self.cfg.data.get("max_confidence", None)
        if min_conf is not None:
            pairs_df = pairs_df[pairs_df["confidence"] >= min_conf]
        if max_conf is not None:
            pairs_df = pairs_df[pairs_df["confidence"] <= max_conf]
        
        if min_conf or max_conf:
            print(f"[PPIDataModule] After confidence filter: {len(pairs_df):,}")
        
        # Split
        seed = self.cfg.training.get("seed", 42)
        train_ratio = self.cfg.training.get("train_val_split", 0.9)
        strategy = self.cfg.training.get("split_strategy", "random")
        
        if strategy == "random":
            print(f"[PPIDataModule] Random split ({train_ratio*100:.0f}% train)")
            train_df, val_df = train_test_split(
                pairs_df, train_size=train_ratio, random_state=seed, shuffle=True
            )
        
        elif strategy == "group":
            print(f"[PPIDataModule] Group split by '{self.split_col}'")
            
            if self.split_col not in pairs_df.columns:
                raise KeyError(f"Split column '{self.split_col}' not found")
            
            groups = pairs_df[self.split_col].unique()
            np.random.seed(seed)
            np.random.shuffle(groups)
            
            val_count = max(1, int(len(groups) * (1 - train_ratio)))
            val_groups = set(groups[:val_count])
            train_groups = set(groups[val_count:])
            
            train_df = pairs_df[pairs_df[self.split_col].isin(train_groups)]
            val_df = pairs_df[pairs_df[self.split_col].isin(val_groups)]
            
            print(f"  Train groups: {len(train_groups):,}, Val groups: {len(val_groups):,}")
        
        else:
            raise ValueError(f"Unknown split_strategy: {strategy}")
        
        print(f"[PPIDataModule] Train: {len(train_df):,}, Val: {len(val_df):,}")
        
        # Create datasets
        balance_clusters = self.cfg.data.get("balance_clusters", False)
        balance_power = self.cfg.data.get("balance_power", 0.5)
        
        self.train_dataset = PPIDataset(
            pairs_df=train_df,
            id2seq=id2seq,
            weight_col=self.weight_col,
            balance_clusters=balance_clusters,
            balance_power=balance_power,
        )
        
        self.val_dataset = PPIDataset(
            pairs_df=val_df,
            id2seq=id2seq,
            balance_clusters=False,
            provided_stats=(self.train_dataset.mean, self.train_dataset.std),
        )
        
        # Sampler
        if self.train_dataset.weights is not None:
            print("[PPIDataModule] Using WeightedRandomSampler")
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
            sampler=self.sampler,
            shuffle=(self.sampler is None),
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )