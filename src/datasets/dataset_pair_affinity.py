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
class PairwiseCollator:
    """ Collates PAIRS of sequences for Ranking. """
    tokenizer: Any
    max_length: int = 1024

    def _tokenize_batch(self, binders: List[str], targets: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        eos = self.tokenizer.eos_token
        cls_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        binder_seqs = [str(b).replace(":", eos) for b in binders]
        target_seqs = [str(t).replace(":", eos) for t in targets]

        b_encoded = self.tokenizer(binder_seqs, add_special_tokens=False)["input_ids"]
        t_encoded = self.tokenizer(target_seqs, add_special_tokens=False)["input_ids"]

        input_ids_list = []
        mask_list = []

        for b_ids, t_ids in zip(b_encoded, t_encoded):
            allowed_len = self.max_length - 3 
            current_len = len(b_ids) + len(t_ids)

            if current_len > allowed_len:
                excess = current_len - allowed_len
                # Priority: Keep Binder, Truncate Target
                if len(t_ids) > excess:
                    t_ids = t_ids[:-excess]
                else:
                    rem = excess - len(t_ids)
                    t_ids = []
                    b_ids = b_ids[:-rem]
            
            full_ids = [cls_id] + b_ids + [eos_id] + t_ids + [eos_id]
            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            mask_list.append(torch.ones(len(full_ids), dtype=torch.long))

        batch_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        batch_mask = pad_sequence(mask_list, batch_first=True, padding_value=0)
        
        return batch_ids, batch_mask

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        better_binders = [x["better_binder"] for x in batch]
        better_targets = [x["better_target"] for x in batch]
        worse_binders = [x["worse_binder"] for x in batch]
        worse_targets = [x["worse_target"] for x in batch]

        b_ids, b_mask = self._tokenize_batch(better_binders, better_targets)
        w_ids, w_mask = self._tokenize_batch(worse_binders, worse_targets)

        return {
            "better_input_ids": b_ids,
            "better_mask": b_mask,
            "worse_input_ids": w_ids,
            "worse_mask": w_mask
        }

class PairwiseAffinityDataset(Dataset):
    def __init__(self, 
                 base_df: pd.DataFrame, 
                 lookup_csv_path: str, 
                 weight_col: Optional[str] = None, 
                 balance_clusters: bool = False,
                 pairs_per_sample: int = 5,
                 min_margin: float = 0.5,       
                 max_margin: Optional[float] = None,
                 max_anchor_val: float = 4.0): # Default set to 4.0
        
        # Load Lookup
        lookup_df = pd.read_csv(lookup_csv_path)
        lookup_df['key'] = lookup_df['type'].astype(str) + "_" + lookup_df['id'].astype(str)
        self.id2seq = dict(zip(lookup_df['key'], lookup_df['seq']))
        
        self.base_df = base_df.copy()

        # 1. Grouping
        if weight_col and weight_col in self.base_df.columns:
            self.stratify_col = weight_col
        else:
            self.stratify_col = "target_id"
        
        # 2. Calculate Weights
        self.cluster_weights = {}
        if balance_clusters:
            counts = self.base_df[self.stratify_col].value_counts()
            self.cluster_weights = (1.0 / np.sqrt(counts)).to_dict()

        # 3. Mining Pairs
        self.pairs = [] 
        self.weights = [] 

        groups = self.base_df.groupby(self.stratify_col)

        for g_name, group in groups:
            # Sort: Best (e.g., -12) -> Worst (e.g., +5)
            # Assuming 'log_Aff' is log Kd, smaller is better.
            valid_group = group.dropna(subset=["log_Aff"]).sort_values("log_Aff", ascending=True)
            aff_values = valid_group["log_Aff"].values 
            n_total = len(valid_group)
            
            if n_total < 2: 
                continue

            cur_weight = self.cluster_weights.get(g_name, 1.0)
            
            # [CRITICAL LOGIC] Only treat sequences with affinity <= max_anchor_val as Anchors
            for i in range(n_total):
                better_val = aff_values[i]
                
                # [NEW CHECK] Stop if the 'better' candidate exceeds the max_anchor_val (4.0)
                # Since array is sorted ascending, all subsequent values are also worse/decoys.
                if better_val > max_anchor_val:
                    break 

                # A. Lower Bound (must be significantly worse than anchor)
                thresh_min = better_val + min_margin
                start_index = np.searchsorted(aff_values, thresh_min, side='right')
                
                # B. Upper Bound
                if max_margin is not None:
                    thresh_max = better_val + max_margin
                    end_index = np.searchsorted(aff_values, thresh_max, side='left')
                else:
                    end_index = n_total
                
                n_candidates = end_index - start_index
                
                if n_candidates > 0:
                    offsets = np.random.randint(start_index, end_index, size=min(n_candidates, pairs_per_sample))
                    
                    row_better = valid_group.iloc[i]
                    
                    for worse_idx in offsets:
                        row_worse = valid_group.iloc[worse_idx]
                        self.pairs.append((row_better, row_worse))
                        self.weights.append(cur_weight)
        
        print(f"[Dataset] Generated {len(self.pairs)} pairs from {len(groups)} groups.")
        if len(self.pairs) > 0:
            print(f"[Dataset] Example Pair: {self.pairs[0][0]['log_Aff']} vs {self.pairs[0][1]['log_Aff']}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row_better, row_worse = self.pairs[idx]
        
        b_key_better = f"binder_{row_better['binder_id']}"
        t_key_better = f"target_{row_better['target_id']}"
        b_key_worse = f"binder_{row_worse['binder_id']}"
        t_key_worse = f"target_{row_worse['target_id']}"

        return {
            "better_binder": self.id2seq.get(b_key_better, ""),
            "better_target": self.id2seq.get(t_key_better, ""),
            "worse_binder": self.id2seq.get(b_key_worse, ""),
            "worse_target": self.id2seq.get(t_key_worse, ""),
        }
    
    def get_weight(self, idx):
        return self.weights[idx]

class PairAffinityDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)
        self.num_workers = cfg.data.num_workers if cfg.data.num_workers is not None else os.cpu_count()
        self.collate_fn = PairwiseCollator(tokenizer=self.tokenizer, max_length=self.cfg.model.max_length)
        
        self.train_dataset = None
        self.val_dataset = None
        self.sampler = None

    def setup(self, stage: Optional[str] = None):
        print(f"[DataModule] Loading base data from {self.cfg.data.base_csv}")
        base_df = pd.read_csv(self.cfg.data.base_csv)
        
        # -----------------------------------------------------------
        # Target-based Split (Prevent Leakage)
        # -----------------------------------------------------------
        all_targets = base_df["target_id"].unique()
        rng = np.random.default_rng(self.cfg.training.seed)
        rng.shuffle(all_targets)
        
        split_idx = int(len(all_targets) * 0.9)
        train_targets = set(all_targets[:split_idx])
        val_targets = set(all_targets[split_idx:])
        
        print(f"[DataModule] Split: {len(train_targets)} Train Targets, {len(val_targets)} Val Targets")
        
        train_df = base_df[base_df["target_id"].isin(train_targets)].copy()
        val_df = base_df[base_df["target_id"].isin(val_targets)].copy()
        
        # -----------------------------------------------------------
        # Create Independent Datasets
        # -----------------------------------------------------------
        # Using 4.0 as default threshold if not in config
        anchor_threshold = self.cfg.data.get("max_anchor_val", 4.0) 

        self.train_dataset = PairwiseAffinityDataset(
            train_df, 
            self.cfg.data.lookup_csv, 
            weight_col=self.cfg.data.get("weight_col"), 
            balance_clusters=self.cfg.data.get("balance_clusters", False),
            pairs_per_sample=self.cfg.data.get("pairs_per_sample", 5),
            min_margin=self.cfg.data.get("min_margin", 0.5),
            max_margin=self.cfg.data.get("max_margin", None),
            max_anchor_val=anchor_threshold 
        )
        
        self.val_dataset = PairwiseAffinityDataset(
            val_df, 
            self.cfg.data.lookup_csv, 
            balance_clusters=False, 
            pairs_per_sample=5, 
            min_margin=0.5,
            max_anchor_val=anchor_threshold 
        )

        if len(self.train_dataset.weights) > 0:
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
            shuffle=(self.sampler is None), 
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.cfg.training.batch_size, 
            collate_fn=self.collate_fn, 
            num_workers=self.num_workers, 
            shuffle=False, 
            pin_memory=True
        )