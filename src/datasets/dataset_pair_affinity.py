import os
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Tuple
from omegaconf import DictConfig

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import LightningDataModule
from transformers import EsmTokenizer

# ----------------------------------------------------------------------
# 1. Pairwise Collator (Supports Concat and Cross-Attn)
# ----------------------------------------------------------------------

@dataclass
class PairwiseCollator:
    """ 
    Collates pairs of sequences. 
    Handles architecture-specific tokenization for ranking tasks.
    """
    tokenizer: Any
    arch: str = "concat"
    max_length: int = 1024

    def _tokenize_batch_concat(self, binders: List[str], targets: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Standard Concat Tokenization: [CLS] Binder [EOS] Target [EOS] """
        eos = self.tokenizer.eos_token
        cls_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        binder_seqs = [str(b).replace(":", eos) for b in binders]
        target_seqs = [str(t).replace(":", eos) for t in targets]

        b_encoded = self.tokenizer(binder_seqs, add_special_tokens=False)["input_ids"]
        t_encoded = self.tokenizer(target_seqs, add_special_tokens=False)["input_ids"]

        input_ids_list, mask_list = [], []
        for b_ids, t_ids in zip(b_encoded, t_encoded):
            allowed_len = self.max_length - 3 
            if len(b_ids) + len(t_ids) > allowed_len:
                # Preserve Binder, truncate Target
                t_ids = t_ids[:max(0, allowed_len - len(b_ids))]
                b_ids = b_ids[:allowed_len]
            
            full_ids = [cls_id] + b_ids + [eos_id] + t_ids + [eos_id]
            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            mask_list.append(torch.ones(len(full_ids), dtype=torch.long))

        return (
            pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id),
            pad_sequence(mask_list, batch_first=True, padding_value=0)
        )

    def _tokenize_batch_cross(self, binders: List[str], targets: List[str]) -> Dict[str, torch.Tensor]:
        """ Cross-Attn Tokenization: Independent Binder and Target tensors """
        b_enc = self.tokenizer(binders, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        t_enc = self.tokenizer(targets, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            "ids": b_enc["input_ids"],
            "mask": b_enc["attention_mask"],
            "target_ids": t_enc["input_ids"],
            "target_mask": t_enc["attention_mask"]
        }

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        better_b, better_t = [x["better_binder"] for x in batch], [x["better_target"] for x in batch]
        worse_b, worse_t = [x["worse_binder"] for x in batch], [x["worse_target"] for x in batch]

        if self.arch == "cross_attn":
            b_data = self._tokenize_batch_cross(better_b, better_t)
            w_data = self._tokenize_batch_cross(worse_b, worse_t)
            return {
                "better_binder_ids": b_data["ids"], "better_binder_mask": b_data["mask"],
                "better_target_ids": b_data["target_ids"], "better_target_mask": b_data["target_mask"],
                "worse_binder_ids": w_data["ids"], "worse_binder_mask": w_data["mask"],
                "worse_target_ids": w_data["target_ids"], "worse_target_mask": w_data["target_mask"],
            }
        else:
            b_ids, b_mask = self._tokenize_batch_concat(better_b, better_t)
            w_ids, w_mask = self._tokenize_batch_concat(worse_b, worse_t)
            return {
                "better_input_ids": b_ids, "better_mask": b_mask,
                "worse_input_ids": w_ids, "worse_mask": w_mask
            }

# ----------------------------------------------------------------------
# 2. Dataset Logic (Maintains your Anchor Filtering)
# ----------------------------------------------------------------------

class PairwiseAffinityDataset(Dataset):
    def __init__(self, base_df, lookup_csv_path, weight_col=None, balance_clusters=False, 
                 pairs_per_sample=5, min_margin=0.5, max_margin=None, max_anchor_val=4.0):
        
        lookup_df = pd.read_csv(lookup_csv_path)
        lookup_df['key'] = lookup_df['type'].astype(str) + "_" + lookup_df['id'].astype(str)
        self.id2seq = dict(zip(lookup_df['key'], lookup_df['seq']))
        self.base_df = base_df.copy()

        stratify_col = weight_col if weight_col in self.base_df.columns else "target_id"
        self.cluster_weights = (1.0 / np.sqrt(self.base_df[stratify_col].value_counts())).to_dict() if balance_clusters else {}

        self.pairs, self.weights = [], []
        for g_name, group in self.base_df.groupby(stratify_col):
            valid_group = group.dropna(subset=["log_Aff"]).sort_values("log_Aff")
            aff_values = valid_group["log_Aff"].values
            if len(valid_group) < 2: continue

            weight = self.cluster_weights.get(g_name, 1.0)
            for i in range(len(valid_group)):
                better_val = aff_values[i]
                if better_val > max_anchor_val: break 

                start = np.searchsorted(aff_values, better_val + min_margin, side='right')
                end = np.searchsorted(aff_values, better_val + (max_margin or 99), side='left') if max_margin else len(aff_values)
                
                if end > start:
                    indices = np.random.randint(start, end, size=min(end-start, pairs_per_sample))
                    row_b = valid_group.iloc[i]
                    for idx in indices:
                        self.pairs.append((row_b, valid_group.iloc[idx]))
                        self.weights.append(weight)

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        rb, rw = self.pairs[idx]
        return {
            "better_binder": self.id2seq.get(f"binder_{rb['binder_id']}", ""),
            "better_target": self.id2seq.get(f"target_{rb['target_id']}", ""),
            "worse_binder": self.id2seq.get(f"binder_{rw['binder_id']}", ""),
            "worse_target": self.id2seq.get(f"target_{rw['target_id']}", ""),
        }

# ----------------------------------------------------------------------
# 3. Pairwise DataModule
# ----------------------------------------------------------------------

class PairAffinityDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)
        arch = self.cfg.model.get("arch", "concat")
        self.collate_fn = PairwiseCollator(tokenizer=self.tokenizer, arch=arch, max_length=self.cfg.model.max_length)
        self.num_workers = cfg.data.get("num_workers", os.cpu_count())
        
        # Determine weight_col once; handle None case
        raw_col = self.cfg.data.get("weight_col")
        self.weight_col = raw_col if raw_col is not None else "target_id"

    def setup(self, stage=None):
        base_df = pd.read_csv(self.cfg.data.base_csv)
        
        # Strict Check
        if self.weight_col not in base_df.columns:
            raise ValueError(f"weight_col '{self.weight_col}' not found in {self.cfg.data.base_csv}")
        
        print(f"[LOG] Splitting and balancing by: {self.weight_col}")

        # Split based on unique groups of weight_col
        all_groups = base_df[self.weight_col].unique()
        np.random.default_rng(self.cfg.training.seed).shuffle(all_groups)
        
        split_idx = int(len(all_groups) * 0.9)
        train_groups = set(all_groups[:split_idx])
        val_groups = set(all_groups[split_idx:])

        train_df = base_df[base_df[self.weight_col].isin(train_groups)]
        val_df = base_df[base_df[self.weight_col].isin(val_groups)]

        self.train_dataset = PairwiseAffinityDataset(
            train_df, self.cfg.data.lookup_csv, weight_col=self.weight_col,
            balance_clusters=self.cfg.data.get("balance_clusters", False),
            pairs_per_sample=self.cfg.data.get("pairs_per_sample", 5),
            min_margin=self.cfg.data.get("min_margin", 0.5),
            max_anchor_val=self.cfg.data.get("max_anchor_val", 4.0)
        )
        self.val_dataset = PairwiseAffinityDataset(val_df, self.cfg.data.lookup_csv, max_anchor_val=4.0)
        
        # Use weights if balancing is enabled and weights were generated
        if self.cfg.data.get("balance_clusters", False) and len(self.train_dataset.weights) > 0:
            self.sampler = WeightedRandomSampler(self.train_dataset.weights, len(self.train_dataset))
        else:
            self.sampler = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.cfg.training.batch_size, 
            collate_fn=self.collate_fn, 
            sampler=self.sampler, 
            shuffle=(self.sampler is None), # Sampler and Shuffle are mutually exclusive
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.cfg.training.batch_size, 
            collate_fn=self.collate_fn, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )