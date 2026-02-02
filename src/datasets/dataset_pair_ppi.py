# src/data/pair_ppi_datamodule.py

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
class PPIPairwiseConcatCollator:
    """Pairwise Concat Collator for PPI ranking."""
    tokenizer: Any
    max_length: int = 512

    def _tokenize_concat(self, seqs_a: List[str], seqs_b: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        
        a_encoded = self.tokenizer(seqs_a, add_special_tokens=False)["input_ids"]
        b_encoded = self.tokenizer(seqs_b, add_special_tokens=False)["input_ids"]
        
        input_ids_list, mask_list = [], []
        
        for a_ids, b_ids in zip(a_encoded, b_encoded):
            allowed = self.max_length - 3
            if len(a_ids) + len(b_ids) > allowed:
                b_ids = b_ids[:max(0, allowed - len(a_ids))]
                a_ids = a_ids[:allowed]
            
            full_ids = [cls_id] + a_ids + [eos_id] + b_ids + [eos_id]
            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            mask_list.append(torch.ones(len(full_ids), dtype=torch.long))
        
        return (
            pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id),
            pad_sequence(mask_list, batch_first=True, padding_value=0)
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        better_a = [x["better_a_seq"] for x in batch]
        better_b = [x["better_b_seq"] for x in batch]
        worse_a = [x["worse_a_seq"] for x in batch]
        worse_b = [x["worse_b_seq"] for x in batch]
        deltas = torch.tensor([x["delta"] for x in batch], dtype=torch.float32)
        
        better_ids, better_mask = self._tokenize_concat(better_a, better_b)
        worse_ids, worse_mask = self._tokenize_concat(worse_a, worse_b)
        
        return {
            "better_input_ids": better_ids,
            "better_mask": better_mask,
            "worse_input_ids": worse_ids,
            "worse_mask": worse_mask,
            "delta": deltas,
        }


@dataclass
class PPIPairwiseCrossAttnCollator:
    """Pairwise Cross-Attention Collator for PPI ranking."""
    tokenizer: Any
    max_length: int = 512

    def _tokenize_batch(self, seqs_a: List[str], seqs_b: List[str]) -> Dict[str, torch.Tensor]:
        a_enc = self.tokenizer(seqs_a, padding=True, truncation=True, 
                               max_length=self.max_length, return_tensors="pt")
        b_enc = self.tokenizer(seqs_b, padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt")
        return {
            "a_ids": a_enc["input_ids"],
            "a_mask": a_enc["attention_mask"],
            "b_ids": b_enc["input_ids"],
            "b_mask": b_enc["attention_mask"],
        }

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        better_a = [x["better_a_seq"] for x in batch]
        better_b = [x["better_b_seq"] for x in batch]
        worse_a = [x["worse_a_seq"] for x in batch]
        worse_b = [x["worse_b_seq"] for x in batch]
        deltas = torch.tensor([x["delta"] for x in batch], dtype=torch.float32)
        
        better_enc = self._tokenize_batch(better_a, better_b)
        worse_enc = self._tokenize_batch(worse_a, worse_b)
        
        return {
            "better_binder_ids": better_enc["a_ids"],
            "better_binder_mask": better_enc["a_mask"],
            "better_target_ids": better_enc["b_ids"],
            "better_target_mask": better_enc["b_mask"],
            "worse_binder_ids": worse_enc["a_ids"],
            "worse_binder_mask": worse_enc["a_mask"],
            "worse_target_ids": worse_enc["b_ids"],
            "worse_target_mask": worse_enc["b_mask"],
            "delta": deltas,
        }


# ----------------------------------------------------------------------
# 2. Dataset
# ----------------------------------------------------------------------

class PPIPairwiseDataset(Dataset):
    """Pairwise ranking dataset from PPI confidence scores."""
    
    def __init__(
        self,
        pairs_df: pd.DataFrame,
        id2seq: Dict[str, str],
        min_margin: float = 0.2,
        pairs_per_anchor: int = 3,
        max_anchors: int = 100000,
        group_col: str = "protein_a",
        split_name: str = "DATASET",
        verbose: bool = True,
    ):
        self.id2seq = id2seq
        self.verbose = verbose
        self.group_col = group_col
        
        self._log(f"\n{'='*25} {split_name} PPI PAIRS {'='*25}")
        self._log(f"[*] Input pairs: {len(pairs_df):,}")
        self._log(f"[*] Grouping by: {group_col}")
        self._log(f"[*] Min margin: {min_margin}")
        
        # Storage
        self.better_a, self.better_b = [], []
        self.worse_a, self.worse_b = [], []
        self.deltas = []
        
        # Generate ranking pairs
        self._generate_pairs(pairs_df, min_margin, pairs_per_anchor, max_anchors)
        
        self._log(f"[*] Total ranking pairs: {len(self):,}")
        self._log(f"{'='*60}\n")
    
    def _generate_pairs(self, df, min_margin, pairs_per_anchor, max_pairs):
        """Generate ranking pairs - vectorized, fast."""
        
        all_better_a, all_better_b = [], []
        all_worse_a, all_worse_b = [], []
        all_deltas = []
        
        skipped_single = 0
        skipped_margin = 0
        
        for anchor_id, group in df.groupby(self.group_col):
            n = len(group)
            if n < 2:
                skipped_single += 1
                continue
            
            group = group.sort_values("confidence")
            confs = group["confidence"].values
            
            # Quick check: any valid pairs?
            if confs[-1] - confs[0] < min_margin:
                skipped_margin += 1
                continue
            
            protein_a = group["protein_a"].values
            protein_b = group["protein_b"].values
            
            # Fast: sample random (i, j) pairs where j > i
            n_possible = n * (n - 1) // 2
            n_sample = min(n_possible, pairs_per_anchor)
            
            count = 0
            attempts = 0
            max_attempts = n_sample * 10
            
            while count < n_sample and attempts < max_attempts:
                i = np.random.randint(0, n - 1)
                j = np.random.randint(i + 1, n)
                
                if confs[j] - confs[i] >= min_margin:
                    all_worse_a.append(protein_a[i])
                    all_worse_b.append(protein_b[i])
                    all_better_a.append(protein_a[j])
                    all_better_b.append(protein_b[j])
                    all_deltas.append(confs[j] - confs[i])
                    count += 1
                
                attempts += 1
        
        # Shuffle and limit
        total = len(all_better_a)
        if total > max_pairs:
            indices = np.random.choice(total, max_pairs, replace=False)
            all_better_a = [all_better_a[i] for i in indices]
            all_better_b = [all_better_b[i] for i in indices]
            all_worse_a = [all_worse_a[i] for i in indices]
            all_worse_b = [all_worse_b[i] for i in indices]
            all_deltas = [all_deltas[i] for i in indices]
        
        self.better_a = all_better_a
        self.better_b = all_better_b
        self.worse_a = all_worse_a
        self.worse_b = all_worse_b
        self.deltas = all_deltas
        
        self._log(f"[*] Skipped (singleton): {skipped_single:,}")
        self._log(f"[*] Skipped (no margin): {skipped_margin:,}")
        self._log(f"[*] Generated {len(self.better_a):,} ranking pairs")
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def __len__(self):
        return len(self.better_a)
    
    def __getitem__(self, idx):
        return {
            "better_a_seq": self.id2seq.get(self.better_a[idx], ""),
            "better_b_seq": self.id2seq.get(self.better_b[idx], ""),
            "worse_a_seq": self.id2seq.get(self.worse_a[idx], ""),
            "worse_b_seq": self.id2seq.get(self.worse_b[idx], ""),
            "delta": self.deltas[idx],
        }


# ----------------------------------------------------------------------
# 3. DataModule
# ----------------------------------------------------------------------

class PairPPIDataModule(LightningDataModule):
    """DataModule for PPI pairwise ranking pretraining."""
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)
        self.num_workers = cfg.data.get("num_workers", 4)
        
        # Collator based on architecture
        arch = cfg.model.get("arch", "concat")
        max_length = cfg.model.get("max_length", 512)
        
        if arch in ["cross_attn", "interaction_map"]:
            print("[PPIDataModule] Using CrossAttnCollator")
            self.collate_fn = PPIPairwiseCrossAttnCollator(
                tokenizer=self.tokenizer, max_length=max_length
            )
        else:
            print("[PPIDataModule] Using ConcatCollator")
            self.collate_fn = PPIPairwiseConcatCollator(
                tokenizer=self.tokenizer, max_length=max_length
            )
        
        self.train_dataset = None
        self.val_dataset = None
        self.sampler = None
    
    @property
    def is_main_process(self) -> bool:
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        return rank == 0
    
    def setup(self, stage: Optional[str] = None):
        # Load data
        pairs_df = pd.read_csv(self.cfg.data.pairs_csv)
        lookup_df = pd.read_csv(self.cfg.data.lookup_csv)
        
        id2seq = dict(zip(lookup_df["id"], lookup_df["seq"]))
        
        if self.is_main_process:
            print(f"[PPIDataModule] Loaded {len(pairs_df):,} pairs, {len(id2seq):,} proteins")
        
        # Filter by source
        sources = self.cfg.data.get("sources", None)
        if sources:
            pairs_df = pairs_df[pairs_df["source"].apply(
                lambda x: any(s in str(x) for s in sources)
            )]
            if self.is_main_process:
                print(f"[PPIDataModule] After source filter ({sources}): {len(pairs_df):,}")
        
        # Filter by confidence
        min_conf = self.cfg.data.get("min_confidence", None)
        max_conf = self.cfg.data.get("max_confidence", None)
        if min_conf is not None:
            pairs_df = pairs_df[pairs_df["confidence"] >= min_conf]
        if max_conf is not None:
            pairs_df = pairs_df[pairs_df["confidence"] <= max_conf]
        
        if self.is_main_process and (min_conf or max_conf):
            print(f"[PPIDataModule] After confidence filter: {len(pairs_df):,}")
        
        # Split
        seed = self.cfg.training.get("seed", 42)
        train_ratio = self.cfg.training.get("train_val_split", 0.9)
        strategy = self.cfg.training.get("split_strategy", "random")
        split_col = self.cfg.data.get("split_col", "protein_a")
        
        if strategy == "random":
            if self.is_main_process:
                print(f"[PPIDataModule] Random split ({train_ratio*100:.0f}% train)")
            train_df, val_df = train_test_split(
                pairs_df, train_size=train_ratio, random_state=seed, shuffle=True
            )
        
        elif strategy == "group":
            if self.is_main_process:
                print(f"[PPIDataModule] Group split by '{split_col}'")
            
            groups = pairs_df[split_col].unique()
            np.random.seed(seed)
            np.random.shuffle(groups)
            
            val_count = max(1, int(len(groups) * (1 - train_ratio)))
            val_groups = set(groups[:val_count])
            train_groups = set(groups[val_count:])
            
            train_df = pairs_df[pairs_df[split_col].isin(train_groups)]
            val_df = pairs_df[pairs_df[split_col].isin(val_groups)]
            
            if self.is_main_process:
                print(f"  Train groups: {len(train_groups):,}, Val groups: {len(val_groups):,}")
        
        else:
            raise ValueError(f"Unknown split_strategy: {strategy}")
        
        if self.is_main_process:
            print(f"[PPIDataModule] Train samples: {len(train_df):,}, Val samples: {len(val_df):,}")
        
        # Dataset params
        min_margin = self.cfg.data.get("min_margin", 0.2)
        pairs_per_anchor = self.cfg.data.get("pairs_per_anchor", 3)
        max_anchors = self.cfg.data.get("max_anchors", 100000)
        group_col = self.cfg.data.get("group_col", "protein_a")
        
        # Create datasets
        self.train_dataset = PPIPairwiseDataset(
            pairs_df=train_df,
            id2seq=id2seq,
            min_margin=min_margin,
            pairs_per_anchor=pairs_per_anchor,
            max_anchors=max_anchors,
            group_col=group_col,
            split_name="TRAIN",
            verbose=self.is_main_process,
        )
        
        self.val_dataset = PPIPairwiseDataset(
            pairs_df=val_df,
            id2seq=id2seq,
            min_margin=min_margin,
            pairs_per_anchor=2,
            max_anchors=max_anchors // 5,
            group_col=group_col,
            split_name="VAL",
            verbose=self.is_main_process,
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
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
