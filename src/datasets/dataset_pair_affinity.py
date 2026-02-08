import os
import torch
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from omegaconf import DictConfig

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pytorch_lightning import LightningDataModule
from transformers import EsmTokenizer

from src.datasets.collators import select_collator, RegressionTestCollator, BinaryClassificationCollator
from src.datasets.test_datasets import TestRegressionDataset, BinaryClassificationTestDataset


# ======================================================================
# HYBRID PAIRWISE DATASET (Target-Centric) - For Train/Val
# ======================================================================
class PairwiseAffinityDataset(Dataset):
    def __init__(self, base_df, lookup_csv_path, pairs_per_sample=2, inter_pps=10, 
                 min_margin=1.0, max_anchors_per_target=2000, split_name="DATASET",
                 verbose=True,
                 add_negatives=False,
                 neg_per_positive=1,
                 neg_log_aff=2.0,
                 pair_within_source=False):  # NEW
        
        self.verbose = verbose
        self.neg_log_aff = neg_log_aff
        self.pair_within_source = pair_within_source  # NEW
        
        lookup_df = pd.read_csv(lookup_csv_path)
        lookup_df['key'] = lookup_df['type'].astype(str) + "_" + lookup_df['id'].astype(str)
        self.id2seq = dict(zip(lookup_df['key'], lookup_df['seq']))
        
        self._log(f"\n{'='*25} {split_name} GENERATION {'='*25}")
        num_df = base_df[base_df['log_Aff'].notna()].copy()
        
        # Check if source column exists when pair_within_source is enabled
        if self.pair_within_source:
            if 'source' not in num_df.columns:
                raise KeyError("'source' column not found but pair_within_source=True")
            self._log(f"[*] Pairing within source: ENABLED")
            self._log(f"[*] Unique sources: {num_df['source'].nunique()}")
        
        counts = num_df.groupby('target_id').size().reset_index(name='pocket_size')
        num_df = num_df.merge(counts, on='target_id')
        self._log(f"[*] Raw Samples: {len(num_df):,}")
        self._log(f"[*] Unique Targets: {num_df['target_id'].nunique():,}")
        
        self.b_better, self.t_better = [], []
        self.b_worse, self.t_worse = [], []
        self.deltas, self.pair_types = [], []
        self.aff_better, self.aff_worse = [], []

        if self.pair_within_source:
            # Generate pairs within each source
            for source_id, source_df in num_df.groupby('source'):
                self._log(f"\n[*] Processing source: {source_id} ({len(source_df)} samples)")
                self._generate_intra_target_pairs(source_df, pairs_per_sample, min_margin, max_anchors_per_target)
                self._generate_inter_target_pairs(source_df, inter_pps)
        else:
            # Original behavior: all data together
            self._generate_intra_target_pairs(num_df, pairs_per_sample, min_margin, max_anchors_per_target)
            self._generate_inter_target_pairs(num_df, inter_pps)

        # --- PATH 3: Negative Pairs ---
        if add_negatives:
            self._add_negative_pairs(num_df, neg_per_positive)

        # Summary
        p_df = pd.DataFrame({'type': self.pair_types})
        self._log(f"\n[*] Pair Breakdown:")
        if self.verbose:
            print(p_df['type'].value_counts().to_string())
        self._log(f"\n[*] TOTAL PAIRS: {len(self.b_better):,}")
        self._log(f"{'='*60}\n")

    def _generate_intra_target_pairs(self, df, pairs_per_sample, min_margin, max_anchors_per_target):
        """Generate intra-target pairs from given dataframe."""
        self._log(f"[*] Generating intra-target pairs...")
        rich_df = df[df['pocket_size'] > 1]
        intra_count = 0
        
        for t_id, group in rich_df.groupby('target_id'):
            group = group.sort_values('log_Aff')
            affs, recs = group['log_Aff'].values, group.to_dict('records')
            n = len(recs)
            starts = np.searchsorted(affs, affs + min_margin, side='right')
            valid_anchors = np.where(starts < n)[0]
            
            if len(valid_anchors) > max_anchors_per_target:
                valid_anchors = np.random.choice(valid_anchors, max_anchors_per_target, replace=False)
            
            if len(valid_anchors) == 0: 
                continue
            
            rand_floats = np.random.random((len(valid_anchors), pairs_per_sample))
            ranges = n - starts[valid_anchors]
            match_indices = (rand_floats * ranges[:, None]).astype(int) + starts[valid_anchors][:, None]
            
            for a_idx, m_idxs in zip(valid_anchors, match_indices):
                for m_idx in m_idxs:
                    if recs[a_idx]['binder_id'] == recs[m_idx]['binder_id']: 
                        continue
                    self._append_pair(recs[a_idx], recs[m_idx], pair_type='intra_target')
                    intra_count += 1
        
        self._log(f"    Intra-target pairs: {intra_count:,}")

    def _generate_inter_target_pairs(self, df, inter_pps):
        """Generate inter-target pairs from given dataframe."""
        self._log(f"[*] Generating inter-target pairs...")
        singleton_df = df[df['pocket_size'] == 1]
        recs = singleton_df.to_dict('records')
        n = len(recs)
        inter_count = 0
        
        if n >= 2:
            for i in range(n):
                competitors = [j for j in range(n) if recs[j]['target_id'] != recs[i]['target_id']]
                if not competitors: 
                    continue
                chosen_idxs = np.random.choice(competitors, min(len(competitors), inter_pps), replace=False)
                for c_idx in chosen_idxs:
                    if recs[i]['log_Aff'] < recs[c_idx]['log_Aff']:
                        self._append_pair(recs[i], recs[c_idx], pair_type='inter_target')
                    else:
                        self._append_pair(recs[c_idx], recs[i], pair_type='inter_target')
                    inter_count += 1
        
        self._log(f"    Inter-target pairs: {inter_count:,}")

    def _add_negative_pairs(self, num_df, neg_per_positive):
        self._log(f"\n[*] Generating negative pairs...")
        self._log(f"    Negatives per target: {neg_per_positive}")
        self._log(f"    Assumed non-binder log_Aff: {self.neg_log_aff}")
        
        target_ids = num_df['target_id'].unique()
        all_binders = num_df['binder_id'].unique()
        n_binders = len(all_binders)
        
        if len(target_ids) < 2:
            self._log(f"    Skipping: need at least 2 targets")
            return
        
        best_idx = num_df.groupby('target_id')['log_Aff'].idxmin()
        target_to_best = {rec['target_id']: rec for rec in num_df.loc[best_idx].to_dict('records')}
        target_to_binders = num_df.groupby('target_id')['binder_id'].apply(set).to_dict()
        
        self._log(f"    Sampling negatives for {len(target_ids):,} targets...")
        neg_count = 0
        
        for tid in target_ids:
            true_rec = target_to_best[tid]
            true_binders = target_to_binders[tid]
            
            sampled = 0
            attempts = 0
            max_attempts = neg_per_positive * 3
            
            while sampled < neg_per_positive and attempts < max_attempts:
                neg_bid = all_binders[np.random.randint(n_binders)]
                attempts += 1
                
                if neg_bid in true_binders:
                    continue
                
                worse_rec = {
                    'binder_id': neg_bid,
                    'target_id': tid,
                    'log_Aff': self.neg_log_aff
                }
                
                self._append_pair(b_rec=true_rec, w_rec=worse_rec, pair_type='negative')
                neg_count += 1
                sampled += 1
        
        self._log(f"    Negative pairs: {neg_count:,}")

    def _append_pair(self, b_rec, w_rec, pair_type: str):
        self.b_better.append(b_rec['binder_id'])
        self.t_better.append(b_rec['target_id'])
        self.b_worse.append(w_rec['binder_id'])
        self.t_worse.append(w_rec['target_id'])
        self.deltas.append(abs(w_rec['log_Aff'] - b_rec['log_Aff']))
        self.pair_types.append(pair_type)
        self.aff_better.append(b_rec['log_Aff'])
        self.aff_worse.append(w_rec['log_Aff'])

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def __len__(self): 
        return len(self.b_better)
    
    def __getitem__(self, idx):
        return {
            "better_binder": self.id2seq.get(f"binder_{self.b_better[idx]}", ""),
            "better_target": self.id2seq.get(f"target_{self.t_better[idx]}", ""),
            "worse_binder": self.id2seq.get(f"binder_{self.b_worse[idx]}", ""),
            "worse_target": self.id2seq.get(f"target_{self.t_worse[idx]}", ""),
            "delta": self.deltas[idx],
            "better_target_id": self.t_better[idx],
            "worse_target_id": self.t_worse[idx],
            "better_log_aff": self.aff_better[idx],
            "worse_log_aff": self.aff_worse[idx],
            "better_binder_id": self.b_better[idx],
            "worse_binder_id": self.b_worse[idx],
        }


# ======================================================================
# DATAMODULE
# ======================================================================
class PairAffinityDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)
        
        self.collate_fn = select_collator(
            cfg.model.get("arch", "concat"), self.tokenizer, cfg.model.max_length, mode="pairwise"
        )
        
        self.num_workers = cfg.data.get("num_workers", 4)
        self.weight_col = self.cfg.data.get("weight_col")
        self.split_col = self.weight_col if self.weight_col is not None else "target_id"
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.test_collate_fn = None
        self.binary_test_dataset = None
        self.binary_test_collate_fn = None
        self.sampler = None

    @property
    def is_main_process(self) -> bool:
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        return rank == 0

    def setup(self, stage=None):
        base_df = pd.read_csv(self.cfg.data.base_csv)
        
        if self.split_col not in base_df.columns:
            raise KeyError(f"Split column '{self.split_col}' not found in CSV.")
        
        seed = self.cfg.training.get("seed", 42)
        train_ratio = self.cfg.training.get("train_val_split", 0.9)
        strategy = self.cfg.training.get("split_strategy", "group")
        
        if self.is_main_process:
            print(f"\n[DataModule] Split/Balance Column: {self.split_col}")
        
        if strategy == "random":
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(
                base_df, train_size=train_ratio, random_state=seed, shuffle=True
            )
            if self.is_main_process:
                print(f"[DataModule] Split Strategy: Random")
            
        elif strategy == "group":
            if self.is_main_process:
                print(f"[DataModule] Split Strategy: Group by {self.split_col}")
            
            unique_groups = base_df[self.split_col].unique()
            counts = base_df[self.split_col].value_counts()
            singleton_groups = counts[counts == 1].index.tolist()
            multi_sample_groups = counts[counts > 1].index.tolist()
            
            if self.is_main_process:
                print(f"  Total unique {self.split_col}s: {len(unique_groups)}")
                print(f"  Singleton {self.split_col}s (1 sample): {len(singleton_groups)}")
                print(f"  Multi-sample {self.split_col}s (>1 sample): {len(multi_sample_groups)}")
            
            np.random.seed(seed)
            np.random.shuffle(multi_sample_groups)
            
            val_group_count = max(1, int(len(multi_sample_groups) * (1 - train_ratio)))
            val_groups = set(multi_sample_groups[:val_group_count])
            train_groups_multi = set(multi_sample_groups[val_group_count:])
            train_groups = train_groups_multi.union(set(singleton_groups))
            
            train_df = base_df[base_df[self.split_col].isin(train_groups)].copy()
            val_df = base_df[base_df[self.split_col].isin(val_groups)].copy()
            
            overlap = train_groups & val_groups
            if len(overlap) > 0:
                raise ValueError(f"ERROR: Train/Val {self.split_col} overlap: {overlap}")
            
            if self.is_main_process:
                print(f"  ✓ Train {self.split_col}s: {len(train_groups)}")
                print(f"  ✓ Val {self.split_col}s: {len(val_groups)} (COMPLETELY UNSEEN)")
        else:
            raise ValueError(f"Unknown split_strategy: {strategy}")
        
        if self.is_main_process:
            print(f"  Train samples: {len(train_df)}, Val samples: {len(val_df)}")

        # Train/Val datasets
        self.train_dataset = PairwiseAffinityDataset(
            train_df, self.cfg.data.lookup_csv, 
            pairs_per_sample=self.cfg.data.get("pairs_per_sample", 2),
            inter_pps=self.cfg.data.get("inter_pps", 10),
            min_margin=self.cfg.data.get("min_margin", 1.0),
            max_anchors_per_target=self.cfg.data.get("max_anchors_per_target", 2000),
            split_name="TRAIN",
            verbose=self.is_main_process,
            add_negatives=self.cfg.data.get("add_negatives", False),
            neg_per_positive=self.cfg.data.get("neg_per_positive", 1),
            neg_log_aff=self.cfg.data.get("neg_log_aff", 2.0),
            pair_within_source=self.cfg.data.get("pair_within_source", False)  # NEW
        )

        self.val_dataset = PairwiseAffinityDataset(
            val_df, self.cfg.data.lookup_csv, 
            pairs_per_sample=2, 
            inter_pps=5, 
            split_name="VAL",
            verbose=self.is_main_process,
            add_negatives=self.cfg.data.get("add_negatives", False),
            neg_per_positive=self.cfg.data.get("neg_per_positive", 1),
            neg_log_aff=self.cfg.data.get("neg_log_aff", 2.0),
            pair_within_source=self.cfg.data.get("pair_within_source", False)  # NEW
        )
        
        # Regression test dataset
        test_csv = self.cfg.data.get("test_csv", None)
        if test_csv and os.path.exists(test_csv):
            if self.is_main_process:
                print(f"\n[DataModule] Loading REGRESSION test set from: {test_csv}")
            
            train_stats_df = train_df[train_df['log_Aff'].notna()]
            train_mean = float(train_stats_df['log_Aff'].mean())
            train_std = float(train_stats_df['log_Aff'].std())
            
            if self.is_main_process:
                print(f"[DataModule] Train normalization stats - Mean: {train_mean:.4f}, Std: {train_std:.4f}")
            
            self.test_dataset = TestRegressionDataset(
                test_csv_path=test_csv,
                provided_stats=(train_mean, train_std),
                verbose=self.is_main_process
            )
            
            self.test_collate_fn = RegressionTestCollator(
                tokenizer=self.tokenizer,
                arch=self.cfg.model.get("arch", "concat"),
                max_length=self.cfg.model.max_length
            )
        else:
            if self.is_main_process:
                print("\n[DataModule] No regression test CSV provided")
            self.test_dataset = None
            self.test_collate_fn = None
        
        # Binary classification test dataset
        binary_test_csv = self.cfg.data.get("binary_test_csv", None)
        if binary_test_csv and os.path.exists(binary_test_csv):
            if self.is_main_process:
                print(f"\n[DataModule] Loading BINARY CLASSIFICATION test set from: {binary_test_csv}")
            
            self.binary_test_dataset = BinaryClassificationTestDataset(
                test_csv_path=binary_test_csv,
                verbose=self.is_main_process
            )
            
            self.binary_test_collate_fn = BinaryClassificationCollator(
                tokenizer=self.tokenizer,
                arch=self.cfg.model.get("arch", "concat"),
                max_length=self.cfg.model.max_length
            )
        else:
            if self.is_main_process:
                print("\n[DataModule] No binary test CSV provided")
            self.binary_test_dataset = None
            self.binary_test_collate_fn = None
        
        self._setup_sampler(base_df)

    def _setup_sampler(self, base_df: pd.DataFrame):
        balance_clusters = self.cfg.data.get("balance_clusters", False)
        balance_power = self.cfg.data.get("balance_power", 0.5)
        
        if balance_clusters:
            if self.is_main_process:
                print(f"\n[Sampler] Balancing enabled with power={balance_power}")
            
            if self.split_col == 'target_id':
                target_to_group = {tid: tid for tid in base_df['target_id'].unique()}
            else:
                target_to_group = base_df.drop_duplicates('target_id').set_index('target_id')[self.split_col].to_dict()
            
            train_groups_in_pairs = []
            for i in range(len(self.train_dataset)):
                t1 = self.train_dataset.t_better[i]
                t2 = self.train_dataset.t_worse[i]
                g1 = target_to_group.get(t1, t1)
                g2 = target_to_group.get(t2, t2)
                train_groups_in_pairs.extend([g1, g2])
            
            group_counts = pd.Series(train_groups_in_pairs).value_counts()
            if self.is_main_process:
                print(f"  Top {self.split_col}s in pairs:")
                print(group_counts.head(5).to_string())
            
            counts_dict = group_counts.to_dict()
            pair_weights = []
            
            for i in range(len(self.train_dataset)):
                t1 = self.train_dataset.t_better[i]
                t2 = self.train_dataset.t_worse[i]
                g1 = target_to_group.get(t1, t1)
                g2 = target_to_group.get(t2, t2)
                combined_freq = np.sqrt(counts_dict.get(g1, 1) * counts_dict.get(g2, 1))
                weight = 1.0 / np.power(combined_freq, balance_power)
                pair_weights.append(weight)
            
            pair_weights = np.array(pair_weights, dtype=np.float32)
            if self.is_main_process:
                print(f"  Weight range: {pair_weights.min():.4f} to {pair_weights.max():.4f}")
            
            self.sampler = WeightedRandomSampler(
                weights=pair_weights, num_samples=len(pair_weights), replacement=True
            )
        else:
            if self.is_main_process:
                print(f"\n[Sampler] Using target-frequency balancing (legacy)")
            t_all = self.train_dataset.t_better + self.train_dataset.t_worse
            target_counts = pd.Series(t_all).value_counts()
            
            counts_dict = target_counts.to_dict()
            pair_weights = []
            for i in range(len(self.train_dataset)):
                t1 = self.train_dataset.t_better[i]
                t2 = self.train_dataset.t_worse[i]
                combined_freq = np.sqrt(counts_dict[t1] * counts_dict[t2])
                pair_weights.append(1.0 / np.sqrt(combined_freq))
            
            self.sampler = WeightedRandomSampler(
                weights=pair_weights, num_samples=len(pair_weights), replacement=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.cfg.training.batch_size, 
            collate_fn=self.collate_fn, sampler=self.sampler, 
            num_workers=self.num_workers, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.cfg.training.batch_size, 
            collate_fn=self.collate_fn, shuffle=False, 
            num_workers=self.num_workers, pin_memory=True
        )
    
    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset, batch_size=self.cfg.training.batch_size,
            collate_fn=self.test_collate_fn, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
    
    def binary_test_dataloader(self):
        if self.binary_test_dataset is None:
            return None
        return DataLoader(
            self.binary_test_dataset, batch_size=self.cfg.training.batch_size,
            collate_fn=self.binary_test_collate_fn, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )