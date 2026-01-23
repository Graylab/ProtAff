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
# 0. REGRESSION COLLATOR (For Test)
# ======================================================================
@dataclass
class RegressionCollator:
    """
    Collator for regression test set - single samples (not pairs).
    """
    tokenizer: Any
    arch: str = "concat"
    max_length: int = 1024

    def _tokenize_concat(self, binders: List[str], targets: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        binders = [x["binder_seq"] for x in batch]
        targets = [x["target_seq"] for x in batch]
        labels = torch.tensor([x["log_Aff"] for x in batch], dtype=torch.float32).unsqueeze(1)

        if self.arch in ["cross_attn", "interaction_map"]:
            b_enc = self.tokenizer(binders, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            t_enc = self.tokenizer(targets, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            return {
                "binder_ids": b_enc["input_ids"],
                "binder_mask": b_enc["attention_mask"],
                "target_ids": t_enc["input_ids"],
                "target_mask": t_enc["attention_mask"],
                "reg_labels": labels
            }
        else:
            input_ids, mask = self._tokenize_concat(binders, targets)
            return {
                "input_ids": input_ids,
                "attention_mask": mask,
                "reg_labels": labels
            }

# ======================================================================
# 1. PAIRWISE COLLATOR (For Train/Val)
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
# 2. HYBRID PAIRWISE DATASET (Target-Centric) - For Train/Val
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
# 3. TEST REGRESSION DATASET (Single Samples) - For Test
# ======================================================================
class TestRegressionDataset(Dataset):
    """
    Test dataset for regression evaluation (single samples, not pairs).
    Expected format: binder_sequence, target_sequence, log_Aff
    """
    def __init__(self, test_csv_path: str, provided_stats: Tuple[float, float]):
        self.test_df = pd.read_csv(test_csv_path)
        
        # Validate required columns
        required_cols = ['binder_sequence', 'target_sequence', 'log_Aff']
        missing = [col for col in required_cols if col not in self.test_df.columns]
        if missing:
            raise KeyError(f"Test CSV missing required columns: {missing}")
        
        self.mean, self.std = provided_stats
        print(f"\n[TestDataset] Using PROVIDED stats. Mean: {self.mean:.4f}, Std: {self.std:.4f}")
        print(f"[TestDataset] Loaded {len(self.test_df)} test samples")
        print(f"[TestDataset] Unique targets: {self.test_df['target_sequence'].nunique()}")
    
    def __len__(self):
        return len(self.test_df)
    
    def __getitem__(self, idx):
        row = self.test_df.iloc[idx]
        raw_aff = float(row["log_Aff"])
        norm_aff = (raw_aff - self.mean) / (self.std + 1e-8)
        
        return {
            "binder_seq": str(row["binder_sequence"]),
            "target_seq": str(row["target_sequence"]),
            "log_Aff": norm_aff
        }


# ======================================================================
# 4. DATAMODULE WITH CLUSTER-BASED SPLITTING
# ======================================================================
class PairAffinityDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)
        
        # Pairwise collator for train/val
        self.collate_fn = PairwiseCollator(
            tokenizer=self.tokenizer, 
            arch=cfg.model.get("arch", "concat"), 
            max_length=cfg.model.max_length
        )
        
        self.num_workers = cfg.data.get("num_workers", 4)
        
        # Use weight_col for splitting (can be target_id or cluster_id)
        self.weight_col = self.cfg.data.get("weight_col")
        self.split_col = self.weight_col if self.weight_col is not None else "target_id"
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.test_collate_fn = None
        self.sampler = None

    def setup(self, stage=None):
        base_df = pd.read_csv(self.cfg.data.base_csv)
        
        if self.split_col not in base_df.columns:
            raise KeyError(f"Split column '{self.split_col}' not found in CSV.")
        
        seed = self.cfg.training.get("seed", 42)
        train_ratio = self.cfg.training.get("train_val_split", 0.9)
        strategy = self.cfg.training.get("split_strategy", "group")
        
        print(f"\n[DataModule] Split/Balance Column: {self.split_col}")
        
        if strategy == "random":
            # Random split (not recommended for generalization)
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(
                base_df, 
                train_size=train_ratio, 
                random_state=seed, 
                shuffle=True
            )
            print(f"[DataModule] Split Strategy: Random")
            
        elif strategy == "group":
            # Group-based split (by target_id or cluster_id)
            print(f"[DataModule] Split Strategy: Group by {self.split_col}")
            
            # Get unique groups
            unique_groups = base_df[self.split_col].unique()
            
            # Count samples per group
            counts = base_df[self.split_col].value_counts()
            singleton_groups = counts[counts == 1].index.tolist()
            multi_sample_groups = counts[counts > 1].index.tolist()
            
            print(f"  Total unique {self.split_col}s: {len(unique_groups)}")
            print(f"  Singleton {self.split_col}s (1 sample): {len(singleton_groups)}")
            print(f"  Multi-sample {self.split_col}s (>1 sample): {len(multi_sample_groups)}")
            
            # Shuffle and split multi-sample groups
            np.random.seed(seed)
            np.random.shuffle(multi_sample_groups)
            
            val_group_count = max(1, int(len(multi_sample_groups) * (1 - train_ratio)))
            val_groups = set(multi_sample_groups[:val_group_count])
            train_groups_multi = set(multi_sample_groups[val_group_count:])
            
            # Add singletons to train
            train_groups = train_groups_multi.union(set(singleton_groups))
            
            # Create splits
            train_df = base_df[base_df[self.split_col].isin(train_groups)].copy()
            val_df = base_df[base_df[self.split_col].isin(val_groups)].copy()
            
            # Verify no overlap
            overlap = train_groups & val_groups
            if len(overlap) > 0:
                raise ValueError(f"ERROR: Train/Val {self.split_col} overlap: {overlap}")
            
            print(f"  ✓ Train {self.split_col}s: {len(train_groups)}")
            print(f"    - Multi-sample: {len(train_groups_multi)}")
            print(f"    - Singletons: {len(singleton_groups)}")
            print(f"  ✓ Val {self.split_col}s: {len(val_groups)} (COMPLETELY UNSEEN)")
            print(f"  ✓ Overlap check: {len(overlap)} (MUST be 0)")
        else:
            raise ValueError(f"Unknown split_strategy: {strategy}")
        
        print(f"  Train samples: {len(train_df)}, Val samples: {len(val_df)}")

        # Create pairwise datasets for train/val
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
            pairs_per_sample=2, 
            inter_pps=5, 
            split_name="VAL"
        )
        
        # Load test dataset (REGRESSION format, not pairs!)
        test_csv = self.cfg.data.get("test_csv", None)
        if test_csv and os.path.exists(test_csv):
            print(f"\n[DataModule] Loading REGRESSION test set from: {test_csv}")
            
            # Get normalization stats from training data (before pair generation)
            train_stats_df = train_df[train_df['log_Aff'].notna()]
            train_mean = float(train_stats_df['log_Aff'].mean())
            train_std = float(train_stats_df['log_Aff'].std())
            
            print(f"[DataModule] Train normalization stats - Mean: {train_mean:.4f}, Std: {train_std:.4f}")
            
            self.test_dataset = TestRegressionDataset(
                test_csv_path=test_csv,
                provided_stats=(train_mean, train_std)
            )
            
            # Create separate collator for test (regression format)
            self.test_collate_fn = RegressionCollator(
                tokenizer=self.tokenizer,
                arch=self.cfg.model.get("arch", "concat"),
                max_length=self.cfg.model.max_length
            )
        else:
            print("\n[DataModule] No test CSV provided - skipping test dataset")
            self.test_dataset = None
            self.test_collate_fn = None
        
        # Weighted sampling based on target/cluster frequency in pairs
        balance_clusters = self.cfg.data.get("balance_clusters", False)
        balance_power = self.cfg.data.get("balance_power", 0.5)
        
        if balance_clusters:
            print(f"\n[Sampler] Balancing enabled with power={balance_power}")
            
            # Get split column values for all targets in pairs
            # Map target_id to split_col value (e.g., cluster_id)
            target_to_group = base_df.set_index('target_id')[self.split_col].to_dict()
            
            # Count frequency of each group in training pairs
            train_groups_in_pairs = []
            for i in range(len(self.train_dataset)):
                t1 = self.train_dataset.t_better[i]
                t2 = self.train_dataset.t_worse[i]
                
                # Get group for each target
                g1 = target_to_group.get(t1, t1)  # Fallback to target_id if not found
                g2 = target_to_group.get(t2, t2)
                
                train_groups_in_pairs.extend([g1, g2])
            
            # Count group occurrences
            group_counts = pd.Series(train_groups_in_pairs).value_counts()
            print(f"  Top {self.split_col}s in pairs:")
            print(group_counts.head(5).to_string())
            
            # Compute weights: 1 / (count(g1) + count(g2))^power
            counts_dict = group_counts.to_dict()
            pair_weights = []
            
            for i in range(len(self.train_dataset)):
                t1 = self.train_dataset.t_better[i]
                t2 = self.train_dataset.t_worse[i]
                
                g1 = target_to_group.get(t1, t1)
                g2 = target_to_group.get(t2, t2)
                
                combined_freq = counts_dict.get(g1, 1) + counts_dict.get(g2, 1)
                weight = 1.0 / np.power(combined_freq, balance_power)
                pair_weights.append(weight)
            
            pair_weights = np.array(pair_weights, dtype=np.float32)
            print(f"  Weight range: {pair_weights.min():.4f} to {pair_weights.max():.4f}")
            print(f"  Weight ratio: {pair_weights.max()/pair_weights.min():.1f}x")
            
            self.sampler = WeightedRandomSampler(
                weights=pair_weights, 
                num_samples=len(pair_weights), 
                replacement=True
            )
        else:
            # Original sampler: balance by target frequency only
            print(f"\n[Sampler] Using target-frequency balancing (legacy)")
            t_all = self.train_dataset.t_better + self.train_dataset.t_worse
            target_counts = pd.Series(t_all).value_counts()
            print(f"  Top Targets in pairs:")
            print(target_counts.head(5).to_string())
            
            counts_dict = target_counts.to_dict()
            pair_weights = []
            for i in range(len(self.train_dataset)):
                t1 = self.train_dataset.t_better[i]
                t2 = self.train_dataset.t_worse[i]
                combined_freq = counts_dict[t1] + counts_dict[t2]
                pair_weights.append(1.0 / np.sqrt(combined_freq))
            
            self.sampler = WeightedRandomSampler(
                weights=pair_weights, 
                num_samples=len(pair_weights), 
                replacement=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.cfg.training.batch_size, 
            collate_fn=self.collate_fn, 
            sampler=self.sampler, 
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
    
    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.training.batch_size,
            collate_fn=self.test_collate_fn,  # Use regression collator!
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )