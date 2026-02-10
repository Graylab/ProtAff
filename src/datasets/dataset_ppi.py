import os
import torch
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pytorch_lightning import LightningDataModule
from transformers import EsmTokenizer
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig

from src.datasets.collators import select_collator
from src.datasets.split_utils import group_split


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

class PPIDataset(Dataset):
    """PPI Dataset for confidence regression. Outputs match shared collator fields."""
    def __init__(
        self,
        pairs_df: pd.DataFrame,
        id2seq: Dict[str, str],
        invert_label: bool = True,
        verbose: bool = True,
    ):
        self.id2seq = id2seq
        self.invert_label = invert_label
        self.verbose = verbose

        self.mean = float(pairs_df["confidence"].mean())
        self.std = float(pairs_df["confidence"].std())

        # Convert to list of dicts for fast __getitem__
        self.data = pairs_df[["protein_a", "protein_b", "confidence"]].to_dict('records')

        self._log(f"[PPIDataset] Loaded {len(self):,} pairs")
        self._log(f"[PPIDataset] Confidence stats - Mean: {self.mean:.4f}, Std: {self.std:.4f}")
        self._log(f"[PPIDataset] Invert label: {self.invert_label}")

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        # Label: invert so lower = better (matches affinity convention)
        if self.invert_label:
            label = 1.0 - row["confidence"]
        else:
            label = row["confidence"]

        return {
            "binder_seq": self.id2seq.get(row["protein_a"], ""),
            "target_seq": self.id2seq.get(row["protein_b"], ""),
            "log_Aff": label,
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

        # Use shared collators (PPI now outputs binder_seq/target_seq/log_Aff)
        arch = cfg.model.get("arch", "concat")
        max_length = cfg.model.get("max_length", 1024)
        self.collate_fn = select_collator(arch, self.tokenizer, max_length, mode="regression")
        print(f"[PPIDataModule] Using {type(self.collate_fn).__name__}")

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

        # Filter rows where either protein is missing from id2seq
        before_len = len(pairs_df)
        has_a = pairs_df["protein_a"].isin(id2seq)
        has_b = pairs_df["protein_b"].isin(id2seq)
        pairs_df = pairs_df[has_a & has_b].copy()
        n_dropped = before_len - len(pairs_df)
        if n_dropped > 0 and self.is_main_process:
            print(f"[PPIDataModule] WARNING: Dropped {n_dropped:,} rows with missing sequences")

        # Split Train/Val
        seed = self.cfg.training.get("seed", 42)
        train_ratio = self.cfg.training.get("train_val_split", 0.95)
        strategy = self.cfg.training.get("split_strategy", "random")

        if strategy == "random":
            if self.is_main_process:
                print(f"[PPIDataModule] Random split ({train_ratio*100:.0f}% train)")
            train_df, val_df = train_test_split(
                pairs_df, train_size=train_ratio, random_state=seed, shuffle=True
            )

        elif strategy == "group":
            split_col = self.cfg.data.get("split_col", "protein_a")
            if self.is_main_process:
                print(f"[PPIDataModule] Group split by '{split_col}'")
            if split_col not in pairs_df.columns:
                raise KeyError(f"Split column '{split_col}' not found")
            train_df, val_df = group_split(
                pairs_df, col=split_col, ratio=train_ratio, seed=seed,
                verbose=self.is_main_process,
            )
        else:
            raise ValueError(f"Unknown split_strategy: {strategy}")

        # Weighted Sampling for Class Imbalance
        use_weighted_sampler = self.cfg.data.get("use_weighted_sampler", True)
        confidence_threshold = self.cfg.data.get("confidence_threshold", 0.5)

        if use_weighted_sampler:
            train_labels = (train_df["confidence"] >= confidence_threshold).astype(int)
            class_counts = train_labels.value_counts().to_dict()

            n_pos = class_counts.get(1, 0)
            n_neg = class_counts.get(0, 0)

            if self.is_main_process:
                print(f"[PPIDataModule] Training Class Distribution (Before Weighting):")
                print(f"  Positive (Binders): {n_pos:,} ({100*n_pos/len(train_df):.1f}%)")
                print(f"  Negative (Non-Binders): {n_neg:,} ({100*n_neg/len(train_df):.1f}%)")

            class_weights = {cls: 1.0 / count for cls, count in class_counts.items() if count > 0}
            sample_weights = train_labels.map(class_weights).values

            self.sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_df),
                replacement=True
            )

            if self.is_main_process:
                print(f"[PPIDataModule] Initialized WeightedRandomSampler (Label-based)")

        # Final summary
        if self.is_main_process:
            print(f"[PPIDataModule] Final Split - Train: {len(train_df):,}, Val: {len(val_df):,}")

        # Create Datasets
        invert_label = self.cfg.data.get("invert_label", True)

        self.train_dataset = PPIDataset(
            pairs_df=train_df,
            id2seq=id2seq,
            invert_label=invert_label,
            verbose=self.is_main_process,
        )

        self.val_dataset = PPIDataset(
            pairs_df=val_df,
            id2seq=id2seq,
            invert_label=invert_label,
            verbose=self.is_main_process,
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