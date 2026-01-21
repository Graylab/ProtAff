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

# ======================================================================
# 1. PAIRWISE COLLATOR
# ======================================================================
@dataclass
class PairwiseCollator:
    tokenizer: Any
    arch: str = "concat"
    max_length: int = 1024

    def _tokenize_batch_concat(self, binders: List[str], targets: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
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
        b_enc = self.tokenizer(binders, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        t_enc = self.tokenizer(targets, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            "ids": b_enc["input_ids"], "mask": b_enc["attention_mask"],
            "target_ids": t_enc["input_ids"], "target_mask": t_enc["attention_mask"]
        }

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        better_b, better_t = [x["better_binder"] for x in batch], [x["better_target"] for x in batch]
        worse_b, worse_t = [x["worse_binder"] for x in batch], [x["worse_target"] for x in batch]
        deltas = torch.tensor([x["delta"] for x in batch], dtype=torch.float32)

        if self.arch in ["cross_attn", "interaction_map"]:
            b_data = self._tokenize_batch_cross(better_b, better_t)
            w_data = self._tokenize_batch_cross(worse_b, worse_t)
            return {
                "better_binder_ids": b_data["ids"], "better_binder_mask": b_data["mask"],
                "better_target_ids": b_data["target_ids"], "better_target_mask": b_data["target_mask"],
                "worse_binder_ids": w_data["ids"], "worse_binder_mask": w_data["mask"],
                "worse_target_ids": w_data["target_ids"], "worse_target_mask": w_data["target_mask"],
                "delta": deltas
            }
        else:
            b_ids, b_mask = self._tokenize_batch_concat(better_b, better_t)
            w_ids, w_mask = self._tokenize_batch_concat(worse_b, worse_t)
            return {
                "better_input_ids": b_ids, "better_mask": b_mask,
                "worse_input_ids": w_ids, "worse_mask": w_mask,
                "delta": deltas
            }

# ======================================================================
# 2. HYBRID PAIRWISE DATASET (Target-Centric)
# ======================================================================
class PairwiseAffinityDataset(Dataset):
    def __init__(self, base_df, lookup_csv_path, pairs_per_sample=2, inter_pps=10, 
                 min_margin=1.0, max_anchors_per_target=2000, split_name="DATASET"):
        
        lookup_df = pd.read_csv(lookup_csv_path)
        lookup_df['key'] = lookup_df['type'].astype(str) + "_" + lookup_df['id'].astype(str)
        self.id2seq = dict(zip(lookup_df['key'], lookup_df['seq']))
        
        print(f"\n{'='*25} {split_name} GENERATION {'='*25}")
        num_df = base_df[base_df['log_Aff'].notna()].copy()
        
        # Identify Rich Pockets vs Singletons
        counts = num_df.groupby('target_id').size().reset_index(name='pocket_size')
        num_df = num_df.merge(counts, on='target_id')
        print(f"[*] Raw Samples: {len(num_df):,}")
        print(f"[*] Unique Targets: {num_df['target_id'].nunique():,}")
        
        self.b_better, self.t_better = [], []
        self.b_worse, self.t_worse = [], []
        self.deltas, self.pair_types = [], []

        # --- PATH 1: Intra-Target (Mutant Ranking) ---
        rich_df = num_df[num_df['pocket_size'] > 1]
        for t_id, group in rich_df.groupby('target_id'):
            group = group.sort_values('log_Aff')
            affs, recs = group['log_Aff'].values, group.to_dict('records')
            n = len(recs)
            starts = np.searchsorted(affs, affs + min_margin, side='right')
            valid_anchors = np.where(starts < n)[0]
            
            if len(valid_anchors) > max_anchors_per_target:
                valid_anchors = np.random.choice(valid_anchors, max_anchors_per_target, replace=False)
            
            if len(valid_anchors) == 0: continue
            
            rand_floats = np.random.random((len(valid_anchors), pairs_per_sample))
            ranges = n - starts[valid_anchors]
            match_indices = (rand_floats * ranges[:, None]).astype(int) + starts[valid_anchors][:, None]
            
            for a_idx, m_idxs in zip(valid_anchors, match_indices):
                for m_idx in m_idxs:
                    if recs[a_idx]['binder_id'] == recs[m_idx]['binder_id']: continue
                    self._append_pair(recs[a_idx], recs[m_idx], is_specificity=False)

        # --- PATH 2: Inter-Target (Global Specificity) ---
        singleton_df = num_df[num_df['pocket_size'] == 1]
        recs = singleton_df.to_dict('records')
        n = len(recs)
        if n >= 2:
            for i in range(n):
                competitors = [j for j in range(n) if recs[j]['target_id'] != recs[i]['target_id']]
                if not competitors: continue
                chosen_idxs = np.random.choice(competitors, min(len(competitors), inter_pps), replace=False)
                for c_idx in chosen_idxs:
                    if recs[i]['log_Aff'] < recs[c_idx]['log_Aff']:
                        self._append_pair(recs[i], recs[c_idx], is_specificity=True)
                    else:
                        self._append_pair(recs[c_idx], recs[i], is_specificity=True)

        p_df = pd.DataFrame({'type': self.pair_types})
        print(f"[*] Breakdown:\n{p_df['type'].value_counts().to_string()}")
        print(f"[*] TOTAL PAIRS: {len(self.b_better):,}")
        print(f"{'='*60}\n")

    def _append_pair(self, b_rec, w_rec, is_specificity=False):
        self.b_better.append(b_rec['binder_id'])
        self.t_better.append(b_rec['target_id'])
        self.b_worse.append(w_rec['binder_id'])
        self.t_worse.append(w_rec['target_id'])
        self.deltas.append(abs(w_rec['log_Aff'] - b_rec['log_Aff']))
        self.pair_types.append('inter_target' if is_specificity else 'intra_target')

    def __len__(self): return len(self.b_better)
    def __getitem__(self, idx):
        return {
            "better_binder": self.id2seq.get(f"binder_{self.b_better[idx]}", ""),
            "better_target": self.id2seq.get(f"target_{self.t_better[idx]}", ""),
            "worse_binder": self.id2seq.get(f"binder_{self.b_worse[idx]}", ""),
            "worse_target": self.id2seq.get(f"target_{self.t_worse[idx]}", ""),
            "delta": self.deltas[idx]
        }

# ======================================================================
# 3. RELIABLE DATAMODULE
# ======================================================================
class PairAffinityDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)
        self.collate_fn = PairwiseCollator(
            tokenizer=self.tokenizer, 
            arch=cfg.model.get("arch", "concat"), 
            max_length=cfg.model.max_length
        )
        self.num_workers = cfg.data.get("num_workers", 4)

    def setup(self, stage=None):
        base_df = pd.read_csv(self.cfg.data.base_csv)
        targets = base_df['target_id'].unique()
        seed = self.cfg.training.get("seed", 42)
        np.random.seed(seed)
        np.random.shuffle(targets)
        
        split_idx = int(len(targets) * 0.9)
        train_targets, val_targets = targets[:split_idx], targets[split_idx:]
        
        print(f"\n[DataModule] Split Strategy: Target-ID Group Split")
        print(f"[DataModule] Train Targets: {len(train_targets)}, Val Targets: {len(val_targets)}")

        train_df = base_df[base_df['target_id'].isin(train_targets)]
        val_df = base_df[base_df['target_id'].isin(val_targets)]

        self.train_dataset = PairwiseAffinityDataset(
            train_df, self.cfg.data.lookup_csv, 
            pairs_per_sample=self.cfg.data.get("pairs_per_sample", 2),
            inter_pps=self.cfg.data.get("inter_pps", 10),
            min_margin=self.cfg.data.get("min_margin", 1.0),
            max_anchors_per_target=self.cfg.data.get("max_anchors_per_target", 2000),
            split_name="TRAIN"
        )
        self.val_dataset = PairwiseAffinityDataset(
            val_df, self.cfg.data.lookup_csv, 
            pairs_per_sample=2, inter_pps=5, split_name="VAL"
        )
        
        # --- THE RELIABLE MATH: W = 1 / sqrt(Count(T1) + Count(T2)) ---
        t_all = self.train_dataset.t_better + self.train_dataset.t_worse
        target_counts = pd.Series(t_all).value_counts()
        print(f"[Sampler] Top Targets Frequency: \n{target_counts.head(5).to_string()}")
        
        counts_dict = target_counts.to_dict()
        pair_weights = []
        for i in range(len(self.train_dataset)):
            t1, t2 = self.train_dataset.t_better[i], self.train_dataset.t_worse[i]
            combined_freq = counts_dict[t1] + counts_dict[t2]
            pair_weights.append(1.0 / np.sqrt(combined_freq))
        
        self.sampler = WeightedRandomSampler(weights=pair_weights, num_samples=len(pair_weights), replacement=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.training.batch_size, 
                          collate_fn=self.collate_fn, sampler=self.sampler, 
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.training.batch_size, 
                          collate_fn=self.collate_fn, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=True)