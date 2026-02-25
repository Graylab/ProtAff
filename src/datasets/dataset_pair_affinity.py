import os
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pytorch_lightning import LightningDataModule
from transformers import EsmTokenizer

from src.datasets.collators import select_collator, RegressionTestCollator, BinaryClassificationCollator
from src.datasets.test_datasets import TestRegressionDataset, BinaryClassificationTestDataset
from src.datasets.split_utils import group_split


# ======================================================================
# HYBRID PAIRWISE DATASET (Target-Centric) - For Train/Val
# ======================================================================
class PairwiseAffinityDataset(Dataset):
    def __init__(self, base_df, lookup_csv_path, pairs_per_sample=2, inter_pps=10,
                 min_margin=1.0, max_margin=None, max_anchors_per_target=2000,
                 pair_group_cols=None, split_name="DATASET",
                 verbose=True, rng=None):

        self.verbose = verbose
        self.rng = rng if rng is not None else np.random.default_rng()

        lookup_df = pd.read_csv(lookup_csv_path)
        lookup_df['key'] = lookup_df['type'].astype(str) + "_" + lookup_df['id'].astype(str)
        self.id2seq = dict(zip(lookup_df['key'], lookup_df['seq']))

        self._split_name = split_name
        self._pairs_per_sample = pairs_per_sample
        self._inter_pps = inter_pps
        self._min_margin = min_margin
        self._max_margin = max_margin
        self._max_anchors_per_target = max_anchors_per_target
        self._pair_group_cols = pair_group_cols

        # Prepare the filtered+merged DataFrame
        num_df = base_df[base_df['log_Aff'].notna()].copy()
        counts = num_df.groupby('target_id').size().reset_index(name='pocket_size')
        num_df = num_df.merge(counts, on='target_id')
        self._num_df = num_df

        self._build_pairs()

    def _build_pairs(self):
        """Generate all pairs from stored DataFrame and parameters."""
        num_df = self._num_df
        split_name = self._split_name

        self._log(f"\n{'='*25} {split_name} GENERATION {'='*25}")
        self._log(f"[*] Raw Samples: {len(num_df):,}")
        self._log(f"[*] Unique Targets: {num_df['target_id'].nunique():,}")

        # Clear all pair lists
        self.b_better, self.t_better = [], []
        self.b_worse, self.t_worse = [], []
        self.deltas, self.pair_types = [], []
        self.dropped_pairs = 0
        self.missing_keys = set()

        if self._pair_group_cols:
            # Mine pairs separately within each group defined by pair_group_cols
            missing_cols = [c for c in self._pair_group_cols if c not in num_df.columns]
            if missing_cols:
                self._log(f"[!] pair_group_cols {missing_cols} not in data, ignoring grouping")
                self._mine_pairs(num_df)
            else:
                self._log(f"[*] Grouping pairs by: {self._pair_group_cols}")
                for group_key, group_df in num_df.groupby(self._pair_group_cols):
                    # Recompute pocket_size within this group
                    grp = group_df.drop(columns='pocket_size')
                    counts = grp.groupby('target_id').size().reset_index(name='pocket_size')
                    grp = grp.merge(counts, on='target_id')
                    self._log(f"  Group {group_key}: {len(grp):,} samples, {grp['target_id'].nunique()} targets")
                    self._mine_pairs(grp)
        else:
            self._mine_pairs(num_df)

        # Summary
        p_df = pd.DataFrame({'type': self.pair_types})
        self._log(f"\n[*] Pair Breakdown:")
        if self.verbose and len(p_df) > 0:
            print(p_df['type'].value_counts().to_string())
        self._log(f"\n[*] TOTAL PAIRS: {len(self.b_better):,}")
        if self.dropped_pairs > 0:
            self._log(f"[!] Dropped pairs (missing sequences): {self.dropped_pairs:,}")
            self._log(f"[!] Missing keys ({len(self.missing_keys)}): {sorted(self.missing_keys)[:20]}")
            if len(self.missing_keys) > 20:
                self._log(f"    ... and {len(self.missing_keys) - 20} more")
        self._log(f"{'='*60}\n")

    def _mine_pairs(self, df):
        """Run intra/inter pair generation on a (sub)dataframe."""
        self._generate_intra_target_pairs(df, self._pairs_per_sample, self._min_margin, self._max_margin, self._max_anchors_per_target)
        self._generate_inter_target_pairs(df, self._inter_pps)

    def _generate_intra_target_pairs(self, df, pairs_per_sample, min_margin, max_margin, max_anchors_per_target):
        """Generate intra-target pairs from given dataframe."""
        self._log(f"[*] Generating intra-target pairs (min_margin={min_margin}, max_margin={max_margin})...")
        rich_df = df[df['pocket_size'] > 1]
        intra_count = 0

        for t_id, group in rich_df.groupby('target_id'):
            group = group.sort_values('log_Aff')
            affs, recs = group['log_Aff'].values, group.to_dict('records')
            n = len(recs)
            starts = np.searchsorted(affs, affs + min_margin, side='right')

            if max_margin is not None:
                ends = np.searchsorted(affs, affs + max_margin, side='right')
            else:
                ends = np.full(n, n, dtype=int)

            valid_anchors = np.where(starts < ends)[0]

            if len(valid_anchors) > max_anchors_per_target:
                valid_anchors = self.rng.choice(valid_anchors, max_anchors_per_target, replace=False)

            if len(valid_anchors) == 0:
                continue

            rand_floats = self.rng.random((len(valid_anchors), pairs_per_sample))
            ranges = ends[valid_anchors] - starts[valid_anchors]
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
            target_ids = np.array([r['target_id'] for r in recs])
            indices = np.arange(n)

            for i in range(n):
                mask = target_ids != target_ids[i]
                competitors = indices[mask]
                if len(competitors) == 0:
                    continue
                chosen_idxs = self.rng.choice(
                    competitors, min(len(competitors), inter_pps), replace=False
                )
                for c_idx in chosen_idxs:
                    if recs[i]['log_Aff'] < recs[c_idx]['log_Aff']:
                        self._append_pair(recs[i], recs[c_idx], pair_type='inter_target')
                    else:
                        self._append_pair(recs[c_idx], recs[i], pair_type='inter_target')
                    inter_count += 1

        self._log(f"    Inter-target pairs: {inter_count:,}")

    def _append_pair(self, b_rec, w_rec, pair_type: str):
        # Filter out pairs where any sequence is missing
        keys = [
            f"binder_{b_rec['binder_id']}", f"target_{b_rec['target_id']}",
            f"binder_{w_rec['binder_id']}", f"target_{w_rec['target_id']}",
        ]
        missing = [k for k in keys if k not in self.id2seq]
        if missing:
            self.dropped_pairs += 1
            self.missing_keys.update(missing)
            return

        self.b_better.append(b_rec['binder_id'])
        self.t_better.append(b_rec['target_id'])
        self.b_worse.append(w_rec['binder_id'])
        self.t_worse.append(w_rec['target_id'])
        self.deltas.append(abs(w_rec['log_Aff'] - b_rec['log_Aff']))
        self.pair_types.append(pair_type)

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
            self.tokenizer, cfg.model.max_length, mode="pairwise"
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

    def setup(self, stage=None):
        base_df = pd.read_csv(self.cfg.data.base_csv)

        if self.split_col not in base_df.columns:
            raise KeyError(f"Split column '{self.split_col}' not found in CSV.")

        seed = self.cfg.training.get("seed", 42)
        train_ratio = self.cfg.training.get("train_val_split", 0.9)
        strategy = self.cfg.training.get("split_strategy", "group")
        rng = np.random.default_rng(seed)

        if strategy == "random":
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(
                base_df, train_size=train_ratio, random_state=seed, shuffle=True
            )

        elif strategy == "group":
            train_df, val_df = group_split(
                base_df, col=self.split_col, ratio=train_ratio, seed=seed,
                verbose=True,
                label_col="log_Aff",
            )
        else:
            raise ValueError(f"Unknown split_strategy: {strategy}")

        # Parse pair_group_cols from config (may be a list or null)
        pair_group_cols = self.cfg.data.get("pair_group_cols", None)
        if pair_group_cols is not None:
            pair_group_cols = list(pair_group_cols)

        # Train/Val datasets
        self.train_dataset = PairwiseAffinityDataset(
            train_df, self.cfg.data.lookup_csv,
            pairs_per_sample=self.cfg.data.get("pairs_per_sample", 2),
            inter_pps=self.cfg.data.get("inter_pps", 10),
            min_margin=self.cfg.data.get("min_margin", 1.0),
            max_margin=self.cfg.data.get("max_margin", None),
            max_anchors_per_target=self.cfg.data.get("max_anchors_per_target", 2000),
            pair_group_cols=pair_group_cols,
            split_name="TRAIN",
            verbose=True,
            rng=rng,
        )

        self.val_dataset = PairwiseAffinityDataset(
            val_df, self.cfg.data.lookup_csv,
            pairs_per_sample=2,
            inter_pps=5,
            min_margin=self.cfg.data.get("min_margin", 1.0),
            max_margin=self.cfg.data.get("max_margin", None),
            pair_group_cols=pair_group_cols,
            split_name="VAL",
            verbose=True,
            rng=rng,
        )

        # Regression test dataset
        test_csv = self.cfg.data.get("test_csv", None)
        if test_csv and os.path.exists(test_csv):
            train_stats_df = train_df[train_df['log_Aff'].notna()]
            train_mean = float(train_stats_df['log_Aff'].mean())
            train_std = float(train_stats_df['log_Aff'].std())

            self.test_dataset = TestRegressionDataset(
                test_csv_path=test_csv,
                provided_stats=(train_mean, train_std),
                verbose=True
            )

            self.test_collate_fn = RegressionTestCollator(
                tokenizer=self.tokenizer,
                max_length=self.cfg.model.max_length
            )
        else:
            self.test_dataset = None
            self.test_collate_fn = None

        # Binary classification test dataset
        binary_test_csv = self.cfg.data.get("binary_test_csv", None)
        if binary_test_csv and os.path.exists(binary_test_csv):
            self.binary_test_dataset = BinaryClassificationTestDataset(
                test_csv_path=binary_test_csv,
                verbose=True
            )

            self.binary_test_collate_fn = BinaryClassificationCollator(
                tokenizer=self.tokenizer,
                max_length=self.cfg.model.max_length
            )
        else:
            self.binary_test_dataset = None
            self.binary_test_collate_fn = None

        self._setup_sampler(base_df)

    def _setup_sampler(self, base_df: pd.DataFrame):
        balance_clusters = self.cfg.data.get("balance_clusters", False)
        balance_power = self.cfg.data.get("balance_power", 0.5)

        if balance_clusters:
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

            self.sampler = WeightedRandomSampler(
                weights=pair_weights, num_samples=len(pair_weights), replacement=True
            )
        else:
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