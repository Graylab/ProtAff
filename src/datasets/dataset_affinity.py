import os
import torch
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from omegaconf import DictConfig

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pytorch_lightning import LightningDataModule
from transformers import EsmTokenizer
from sklearn.model_selection import train_test_split

from src.datasets.collators import select_collator, BinaryClassificationCollator
from src.datasets.test_datasets import TestRegressionDataset, BinaryClassificationTestDataset
from src.datasets.split_utils import group_split

# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

class AffinityDataset(Dataset):
    def __init__(self, 
                 base_df: pd.DataFrame, 
                 lookup_csv_path: str, 
                 weight_col: Optional[str] = None, 
                 balance_clusters: bool = False,
                 balance_power: float = 0.5,
                 provided_stats: Optional[Tuple[float, float]] = None):
        
        lookup_df = pd.read_csv(lookup_csv_path)
        lookup_df['key'] = lookup_df['type'].astype(str) + "_" + lookup_df['id'].astype(str)
        self.id2seq = dict(zip(lookup_df['key'], lookup_df['seq']))
        
        self.base_df = base_df.copy()
        self.base_df["binder_key"] = "binder_" + self.base_df["binder_id"].astype(str)
        self.base_df["target_key"] = "target_" + self.base_df["target_id"].astype(str)
        
        self.weights = None
        if balance_clusters:
            # Use weight_col if provided, otherwise default to target_id
            stratify_col = weight_col if (weight_col and weight_col in self.base_df.columns) else "target_id"
            counts = self.base_df[stratify_col].value_counts()
            
            # DIAGNOSTIC: Print imbalance stats
            print(f"\n[Dataset] Imbalance Statistics for '{stratify_col}':")
            print(f"  Total unique {stratify_col}s: {len(counts)}")
            print(f"  Max samples per {stratify_col}: {counts.max()}")
            print(f"  Min samples per {stratify_col}: {counts.min()}")
            print(f"  Median samples: {counts.median()}")
            print(f"  Mean samples: {counts.mean():.1f}")
            print(f"  Imbalance ratio (max/min): {counts.max()/counts.min():.1f}x")
            
            freqs = self.base_df[stratify_col].map(counts)
            
            # Use configurable power for balancing strength
            self.weights = (1.0 / np.power(freqs, balance_power)).to_numpy(dtype=np.float32)
            
            # DIAGNOSTIC: Print weight stats
            print(f"  Balance power: {balance_power}")
            print(f"  Weight range: {self.weights.min():.4f} to {self.weights.max():.4f}")
            print(f"  Weight ratio (max/min): {self.weights.max()/self.weights.min():.1f}x")

        self.data = self.base_df[["binder_key", "target_key", "log_Aff"]].to_dict('records')

        if provided_stats:
            self.mean, self.std = provided_stats
            print(f"[Dataset] Using PROVIDED stats. Mean: {self.mean:.4f}, Std: {self.std:.4f}")
        else:
            all_labels = self.base_df["log_Aff"].values
            self.mean = float(np.mean(all_labels))
            self.std = float(np.std(all_labels))
            print(f"[Dataset] Calculated internal stats. Mean: {self.mean:.4f}, Std: {self.std:.4f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        raw_aff = float(row["log_Aff"])
        # Negate so predicted_affinity is higher = better (log_Kd is lower = better)
        norm_aff = -((raw_aff - self.mean) / (self.std + 1e-8))

        return {
            "binder_seq": self.id2seq.get(row["binder_key"], ""),
            "target_seq": self.id2seq.get(row["target_key"], ""),
            "log_Aff": norm_aff,
        }


# ----------------------------------------------------------------------
# DataModule
# ----------------------------------------------------------------------
class AffinityDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)
        self.num_workers = cfg.data.num_workers if cfg.data.num_workers is not None else os.cpu_count()
        
        self.collate_fn = select_collator(self.tokenizer, self.cfg.model.max_length, mode="regression")
        print(f"[DataModule] Initializing {type(self.collate_fn).__name__}")
        
        self.weight_col = self.cfg.data.get("weight_col")
        self.split_col = self.weight_col if self.weight_col is not None else "target_id"
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.binary_test_dataset = None
        self.binary_test_collate_fn = None
        self.sampler = None

    def setup(self, stage: Optional[str] = None):
        print(f"[DataModule] Split/Balance Column: {self.split_col}")
        base_df = pd.read_csv(self.cfg.data.base_csv)
        
        if self.split_col not in base_df.columns:
            raise KeyError(f"Split column '{self.split_col}' not found in CSV.")

        train_ratio = self.cfg.training.get("train_val_split", 0.9)
        strategy = self.cfg.training.get("split_strategy", "random")
        seed = self.cfg.training.seed

        if strategy == "random":
            print(f"--- Running RANDOM Split ({train_ratio*100}% Train) ---")
            train_df, val_df = train_test_split(
                base_df,
                train_size=train_ratio,
                random_state=seed,
                shuffle=True
            )
        
        elif strategy == "group":
            print(f"--- Running STRICT GROUP Split (Val has UNSEEN {self.split_col}s) ---")
            train_df, val_df = group_split(
                base_df, col=self.split_col, ratio=train_ratio, seed=seed, verbose=True
            )

        else:
            raise ValueError(f"Unknown split_strategy: {strategy}")

        print(f"Final Sample Counts -> Train: {len(train_df)}, Val: {len(val_df)}")

        # Get balance_power from config
        balance_power = self.cfg.data.get("balance_power", 0.25)
        
        # 3. Construct Train/Val Datasets
        self.train_dataset = AffinityDataset(
            train_df, 
            self.cfg.data.lookup_csv, 
            weight_col=self.weight_col,  # Pass weight_col for balancing
            balance_clusters=self.cfg.data.get("balance_clusters", False),
            balance_power=balance_power
        )
        
        self.val_dataset = AffinityDataset(
            val_df, 
            self.cfg.data.lookup_csv, 
            weight_col=self.weight_col,  # Pass weight_col for consistency
            balance_clusters=False,  # NEVER balance validation!
            provided_stats=(self.train_dataset.mean, self.train_dataset.std)
        )

        # 4. Construct Test Dataset (if provided)
        test_csv = self.cfg.data.get("test_csv", None)
        if test_csv and os.path.exists(test_csv):
            print(f"[DataModule] Loading test set from: {test_csv}")
            self.test_dataset = TestRegressionDataset(
                test_csv_path=test_csv,
                provided_stats=(self.train_dataset.mean, self.train_dataset.std)
            )
        else:
            print("[DataModule] No test CSV provided or file not found - skipping test dataset")
            self.test_dataset = None
        
        # Binary classification test dataset
        binary_test_csv = self.cfg.data.get("binary_test_csv", None)
        if binary_test_csv and os.path.exists(binary_test_csv):
            print(f"[DataModule] Loading BINARY CLASSIFICATION test set from: {binary_test_csv}")
            self.binary_test_dataset = BinaryClassificationTestDataset(
                test_csv_path=binary_test_csv
            )
            self.binary_test_collate_fn = BinaryClassificationCollator(
                tokenizer=self.tokenizer,
                max_length=self.cfg.model.max_length,
            )
        else:
            print("[DataModule] No binary test CSV provided")
            self.binary_test_dataset = None
            self.binary_test_collate_fn = None

        # 5. Setup weighted sampler if balancing enabled
        if self.train_dataset.weights is not None:
            print(f"[DataModule] ✓ Using WeightedRandomSampler for {len(self.train_dataset.weights)} samples")
            self.sampler = WeightedRandomSampler(
                weights=self.train_dataset.weights, 
                num_samples=len(self.train_dataset), 
                replacement=True
            )
        else:
            print(f"[DataModule] ✓ Using standard random shuffling (no sample weighting)")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.cfg.training.batch_size, 
            collate_fn=self.collate_fn, num_workers=self.num_workers, 
            sampler=self.sampler, shuffle=(self.sampler is None), pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.cfg.training.batch_size, 
            collate_fn=self.collate_fn, num_workers=self.num_workers, 
            shuffle=False, pin_memory=True
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.training.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
    
    def binary_test_dataloader(self):
        if self.binary_test_dataset is None:
            return None
        return DataLoader(
            self.binary_test_dataset,
            batch_size=self.cfg.training.batch_size,
            collate_fn=self.binary_test_collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )