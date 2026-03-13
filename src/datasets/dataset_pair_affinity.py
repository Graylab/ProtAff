import json
import os
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pytorch_lightning import LightningDataModule
from transformers import EsmTokenizer

from src.datasets.collators import select_collator, BinaryClassificationCollator
from src.datasets.test_datasets import TestRegressionDataset, BinaryClassificationTestDataset
from src.datasets.split_utils import group_split, within_group_split


# ======================================================================
# HYBRID PAIRWISE DATASET (Target-Centric) - For Train/Val
# ======================================================================
class PairwiseAffinityDataset(Dataset):
    def __init__(self, base_df, lookup_csv_path, intra_pps=2, inter_pps=10,
                 min_margin=1.0, max_margin=None, intra_max_anchors_per_target=2000,
                 pair_group_cols=None,
                 inter_samples_per_target=None, split_name="DATASET",
                 verbose=True, rng=None):

        self.verbose = verbose
        self.rng = rng if rng is not None else np.random.default_rng()

        lookup_df = pd.read_csv(lookup_csv_path)
        lookup_df['key'] = lookup_df['type'].astype(str) + "_" + lookup_df['id'].astype(str)
        self.id2seq = dict(zip(lookup_df['key'], lookup_df['seq']))

        self._split_name = split_name
        self._intra_pps = intra_pps
        self._inter_pps = inter_pps
        self._min_margin = min_margin
        self._max_margin = max_margin
        self._intra_max_anchors_per_target = intra_max_anchors_per_target
        self._pair_group_cols = pair_group_cols
        self._inter_samples_per_target = inter_samples_per_target

        # Prepare the filtered+merged DataFrame
        num_df = base_df[base_df['log_Aff'].notna()].copy()
        counts = num_df.groupby('target_id').size().reset_index(name='pocket_size')
        num_df = num_df.merge(counts, on='target_id')
        self._num_df = num_df

        self._loss_type = None  # set externally if needed
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
        self.sources = []
        self.lambda_weights = []
        self.dropped_pairs = 0
        self.missing_keys = set()
        self._current_group = None

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
                    self._current_group = group_key
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
        self._generate_intra_target_pairs(df, self._intra_pps, self._min_margin, self._max_margin, self._intra_max_anchors_per_target)
        if self._inter_samples_per_target is not None:
            self._generate_inter_target_pairs(df, self._inter_pps,
                                              self._min_margin, self._max_margin,
                                              self._inter_samples_per_target)
        else:
            self._generate_inter_target_pairs_legacy(df, self._inter_pps)

    def _generate_intra_target_pairs(self, df, intra_pps, min_margin, max_margin, max_anchors_per_target):
        """Generate intra-target pairs from given dataframe."""
        self._log(f"[*] Generating intra-target pairs (min_margin={min_margin}, max_margin={max_margin})...")
        rich_df = df[df['pocket_size'] > 1]
        intra_count = 0

        for t_id, group in rich_df.groupby('target_id'):
            group = group.sort_values('log_Aff', kind='mergesort')
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

            rand_floats = self.rng.random((len(valid_anchors), intra_pps))
            ranges = ends[valid_anchors] - starts[valid_anchors]
            match_indices = (rand_floats * ranges[:, None]).astype(int) + starts[valid_anchors][:, None]

            for a_idx, m_idxs in zip(valid_anchors, match_indices):
                for m_idx in m_idxs:
                    if recs[a_idx]['binder_id'] == recs[m_idx]['binder_id']:
                        continue
                    self._append_pair(recs[a_idx], recs[m_idx], pair_type='intra_target')
                    intra_count += 1

        self._log(f"    Intra-target pairs: {intra_count:,}")

    def _generate_inter_target_pairs_legacy(self, df, inter_pps):
        """Generate inter-target pairs from singleton targets only (legacy behavior)."""
        self._log(f"[*] Generating inter-target pairs (legacy, singletons only)...")
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

    def _generate_inter_target_pairs(self, df, inter_pps, min_margin, max_margin, samples_per_target):
        """Generate inter-target pairs with margin filtering using searchsorted.

        For each target, sample up to samples_per_target binders. Sort all sampled
        records by log_Aff, then use searchsorted to find pairs within the margin window
        across different targets.
        """
        self._log(f"[*] Generating inter-target pairs (margin=[{min_margin}, {max_margin}], "
                  f"samples_per_target={samples_per_target})...")

        # Sample up to samples_per_target binders per target
        sampled_recs = []
        for t_id, group in df.groupby('target_id'):
            recs = group.to_dict('records')
            if len(recs) <= samples_per_target:
                sampled_recs.extend(recs)
            else:
                chosen = self.rng.choice(len(recs), samples_per_target, replace=False)
                sampled_recs.extend(recs[i] for i in chosen)

        if len(sampled_recs) < 2:
            self._log(f"    Inter-target pairs: 0")
            return

        # Sort by log_Aff ascending (better binders first)
        sampled_recs.sort(key=lambda r: r['log_Aff'])
        affs = np.array([r['log_Aff'] for r in sampled_recs])
        target_ids = np.array([r['target_id'] for r in sampled_recs])
        n = len(sampled_recs)

        # searchsorted for margin window: for anchor i, find candidates j where
        # affs[j] in [affs[i] + min_margin, affs[i] + max_margin]
        starts = np.searchsorted(affs, affs + min_margin, side='left')
        if max_margin is not None:
            ends = np.searchsorted(affs, affs + max_margin, side='right')
        else:
            ends = np.full(n, n, dtype=int)

        inter_count = 0
        for i in range(n):
            if starts[i] >= ends[i]:
                continue
            # Filter to different target_ids within the window
            window = np.arange(starts[i], ends[i])
            diff_target_mask = target_ids[window] != target_ids[i]
            candidates = window[diff_target_mask]
            if len(candidates) == 0:
                continue

            chosen = candidates if len(candidates) <= inter_pps else \
                self.rng.choice(candidates, inter_pps, replace=False)

            for c_idx in chosen:
                # Anchor i has lower log_Aff (better), candidate c_idx has higher (worse)
                self._append_pair(sampled_recs[i], sampled_recs[c_idx], pair_type='inter_target')
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
        self.sources.append(self._current_group)
        self.lambda_weights.append(1.0)  # default; overwritten by _compute_lambda_weights

    def _compute_lambda_weights(self):
        """Compute |ΔNDCG| weights for intra-target pairs."""
        num_df = self._num_df

        # Build per-target ideal DCG and position/gain lookups
        target_info = {}  # target_id -> {idealDCG, binder_rank, binder_gain}
        for t_id, group in num_df.groupby('target_id'):
            affs = group['log_Aff'].values
            binder_ids = group['binder_id'].values

            # Normalize affinities to [0, 4] range within this target
            aff_min, aff_max = affs.min(), affs.max()
            if aff_max > aff_min:
                norm_affs = (affs - aff_min) / (aff_max - aff_min) * 4.0
            else:
                norm_affs = np.full_like(affs, 2.0)

            gains = 2.0 ** norm_affs - 1.0

            # Sort by gain descending to get ideal ranking
            sorted_idx = np.argsort(-gains)
            ideal_dcg = np.sum(gains[sorted_idx] / np.log2(np.arange(len(sorted_idx)) + 2))

            # Map binder_id -> (ideal_rank_position, gain)
            binder_rank = {}
            binder_gain = {}
            for rank, idx in enumerate(sorted_idx):
                bid = binder_ids[idx]
                binder_rank[bid] = rank
                binder_gain[bid] = gains[idx]

            target_info[t_id] = {
                'idealDCG': ideal_dcg,
                'binder_rank': binder_rank,
                'binder_gain': binder_gain,
            }

        # Compute lambda weight for each pair
        updated = 0
        for i in range(len(self.lambda_weights)):
            if self.pair_types[i] != 'intra_target':
                continue  # inter-target pairs keep weight=1.0

            t_id = self.t_better[i]
            info = target_info.get(t_id)
            if info is None or info['idealDCG'] < 1e-10:
                continue

            bid_better = self.b_better[i]
            bid_worse = self.b_worse[i]

            gain_b = info['binder_gain'].get(bid_better)
            gain_w = info['binder_gain'].get(bid_worse)
            rank_b = info['binder_rank'].get(bid_better)
            rank_w = info['binder_rank'].get(bid_worse)

            if gain_b is None or gain_w is None or rank_b is None or rank_w is None:
                continue

            delta_gain = abs(gain_b - gain_w)
            delta_discount = abs(1.0 / np.log2(rank_b + 2) - 1.0 / np.log2(rank_w + 2))
            delta_ndcg = delta_gain * delta_discount / info['idealDCG']

            self.lambda_weights[i] = max(delta_ndcg, 1e-6)
            updated += 1

        self._log(f"[*] LambdaRank: computed ΔNDCG weights for {updated:,} intra-target pairs")

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def __len__(self): 
        return len(self.b_better)
    
    def __getitem__(self, idx):
        item = {
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
        if self._loss_type == "lambdarank":
            item["lambda_weight"] = self.lambda_weights[idx]
        return item


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

        split_file = self.cfg.data.get("split_file", None)
        if split_file and os.path.exists(split_file):
            with open(split_file) as f:
                split_info = json.load(f)
            if split_info["split_col"] != self.split_col:
                raise ValueError(
                    f"split_file split_col '{split_info['split_col']}' != "
                    f"datamodule split_col '{self.split_col}'"
                )
            val_groups = set(split_info["val_groups"])
            val_df = base_df[base_df[self.split_col].isin(val_groups)].copy()
            train_df = base_df[~base_df[self.split_col].isin(val_groups)].copy()
            print(f"[Split] Loaded from {split_file}: "
                  f"{len(train_df):,} train / {len(val_df):,} val samples, "
                  f"{len(val_groups)} val groups")
        elif strategy == "random":
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
        elif strategy == "within_group":
            train_df, val_df = within_group_split(
                base_df, col=self.split_col, ratio=train_ratio, seed=seed,
                verbose=True,
            )
        else:
            raise ValueError(f"Unknown split_strategy: {strategy}")

        # Parse pair_group_cols from config (may be a list or null)
        pair_group_cols = self.cfg.data.get("pair_group_cols", None)
        if pair_group_cols is not None:
            pair_group_cols = list(pair_group_cols)

        # Parse inter-target params
        inter_samples_per_target = self.cfg.data.get("inter_samples_per_target", None)

        # Train/Val datasets
        loss_type = self.cfg.training.get("loss_type", "margin")

        self.train_dataset = PairwiseAffinityDataset(
            train_df, self.cfg.data.lookup_csv,
            intra_pps=self.cfg.data.get("intra_pps", 2),
            inter_pps=self.cfg.data.get("inter_pps", 10),
            min_margin=self.cfg.data.get("min_margin", 1.0),
            max_margin=self.cfg.data.get("max_margin", None),
            intra_max_anchors_per_target=self.cfg.data.get("intra_max_anchors_per_target", 2000),
            pair_group_cols=pair_group_cols,
            inter_samples_per_target=inter_samples_per_target,
            split_name="TRAIN",
            verbose=True,
            rng=rng,
        )
        self.train_dataset._loss_type = loss_type
        if loss_type == "lambdarank":
            self.train_dataset._compute_lambda_weights()

        self.val_dataset = PairwiseAffinityDataset(
            val_df, self.cfg.data.lookup_csv,
            intra_pps=2,
            inter_pps=5,
            min_margin=self.cfg.data.get("min_margin", 1.0),
            max_margin=self.cfg.data.get("max_margin", None),
            pair_group_cols=pair_group_cols,
            inter_samples_per_target=inter_samples_per_target,
            split_name="VAL",
            verbose=True,
            rng=rng,
        )
        self.val_dataset._loss_type = loss_type
        if loss_type == "lambdarank":
            self.val_dataset._compute_lambda_weights()

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

            self.test_collate_fn = select_collator(
                tokenizer=self.tokenizer,
                max_length=self.cfg.model.max_length,
                mode="regression_test"
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

            pair_weights = np.array(pair_weights, dtype=np.float32)

        # Optional source-aware balancing
        balance_sources = self.cfg.data.get("balance_sources", False)
        if balance_sources and any(s is not None for s in self.train_dataset.sources):
            source_series = pd.Series(self.train_dataset.sources)
            source_counts = source_series.value_counts().to_dict()
            source_weights = np.array(
                [1.0 / source_counts.get(s, 1) for s in self.train_dataset.sources],
                dtype=np.float32,
            )
            # Normalize so mean source weight is 1 (preserves scale of existing weights)
            source_weights /= source_weights.mean()
            pair_weights *= source_weights

        g = torch.Generator()
        g.manual_seed(self.cfg.training.get("seed", 42))
        self.sampler = WeightedRandomSampler(
            weights=pair_weights, num_samples=len(pair_weights), replacement=True,
            generator=g,
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