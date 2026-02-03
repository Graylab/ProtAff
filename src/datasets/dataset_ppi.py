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
class PPIConcatCollator:
    """Concat Collator: [CLS] seq_a [EOS] seq_b [EOS]"""
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        seqs_a = [str(f["seq_a"]) for f in features]
        seqs_b = [str(f["seq_b"]) for f in features]
        
        a_encoded = self.tokenizer(seqs_a, add_special_tokens=False)["input_ids"]
        b_encoded = self.tokenizer(seqs_b, add_special_tokens=False)["input_ids"]
        
        cls_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        
        input_ids_list, mask_list = [], []
        
        for a_ids, b_ids in zip(a_encoded, b_encoded):
            allowed = self.max_length - 3
            if len(a_ids) + len(b_ids) > allowed:
                b_ids = b_ids[:max(0, allowed - len(a_ids))]
                a_ids = a_ids[:allowed]
            
            full_ids = [cls_id] + a_ids + [eos_id] + b_ids + [eos_id]
            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            mask_list.append(torch.ones(len(full_ids), dtype=torch.long))
        
        labels = torch.tensor([f["label"] for f in features], dtype=torch.float32)
        
        return {
            "input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id),
            "attention_mask": pad_sequence(mask_list, batch_first=True, padding_value=0),
            "labels": labels,
        }


@dataclass
class PPICrossAttnCollator:
    """Cross-Attention Collator: Separate tensors for seq_a and seq_b."""
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        seqs_a = [str(f["seq_a"]) for f in features]
        seqs_b = [str(f["seq_b"]) for f in features]
        
        a_enc = self.tokenizer(
            seqs_a, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        b_enc = self.tokenizer(
            seqs_b, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        
        labels = torch.tensor([f["label"] for f in features], dtype=torch.float32)
        
        return {
            "binder_ids": a_enc["input_ids"],
            "binder_mask": a_enc["attention_mask"],
            "target_ids": b_enc["input_ids"],
            "target_mask": b_enc["attention_mask"],
            "labels": labels,
        }


# ----------------------------------------------------------------------
# 2. Dataset
# ----------------------------------------------------------------------

class PPIDataset(Dataset):
    """PPI Dataset for confidence regression."""
    
    def __init__(
        self,
        pairs_df: pd.DataFrame,
        id2seq: Dict[str, str],
        balance_clusters: bool = False,
        balance_power: float = 0.5,
        provided_stats: Optional[Tuple[float, float]] = None,
        invert_label: bool = True,
        verbose: bool = True,
    ):
        self.id2seq = id2seq
        self.pairs_df = pairs_df.copy()
        self.invert_label = invert_label
        self.verbose = verbose
        
        # Weights for sampling
        self.weights = None
        if balance_clusters:
            # Count frequency of each protein
            protein_a_counts = self.pairs_df["protein_a"].value_counts()
            protein_b_counts = self.pairs_df["protein_b"].value_counts()
            
            self._log(f"\n[PPIDataset] Balancing by protein frequency:")
            self._log(f"  Unique protein_a: {len(protein_a_counts):,}")
            self._log(f"  Unique protein_b: {len(protein_b_counts):,}")
            self._log(f"  protein_a freq range: {protein_a_counts.min()} - {protein_a_counts.max()}")
            self._log(f"  protein_b freq range: {protein_b_counts.min()} - {protein_b_counts.max()}")
            
            # Get frequencies for each pair
            freq_a = self.pairs_df["protein_a"].map(protein_a_counts).values.astype(np.float32)
            freq_b = self.pairs_df["protein_b"].map(protein_b_counts).values.astype(np.float32)
            
            # Geometric mean of frequencies
            combined_freq = np.sqrt(freq_a * freq_b)
            
            # Weight = 1 / freq^power (no normalization needed)
            self.weights = (1.0 / np.power(combined_freq, balance_power)).astype(np.float32)
            
            self._log(f"  Balance power: {balance_power}")
            self._log(f"  Weight range: {self.weights.min():.4f} - {self.weights.max():.4f}")
            self._log(f"  Weight ratio (max/min): {self.weights.max() / self.weights.min():.1f}x")
        
        # Stats for normalization
        if provided_stats:
            self.mean, self.std = provided_stats
        else:
            self.mean = float(self.pairs_df["confidence"].mean())
            self.std = float(self.pairs_df["confidence"].std())
        
        self._log(f"[PPIDataset] Loaded {len(self):,} pairs")
        self._log(f"[PPIDataset] Confidence stats - Mean: {self.mean:.4f}, Std: {self.std:.4f}")
        self._log(f"[PPIDataset] Invert label: {self.invert_label}")
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def __len__(self):
        return len(self.pairs_df)
    
    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        
        # Label: invert so lower = better (matches affinity convention)
        if self.invert_label:
            label = 1.0 - row["confidence"]
        else:
            label = row["confidence"]
        
        return {
            "seq_a": self.id2seq.get(row["protein_a"], ""),
            "seq_b": self.id2seq.get(row["protein_b"], ""),
            "label": label,
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
        
        # Select collator
        arch = cfg.model.get("arch", "concat")
        max_length = cfg.model.get("max_length", 512)
        
        if arch in ["cross_attn", "interaction_map"]:
            print("[PPIDataModule] Using CrossAttnCollator")
            self.collate_fn = PPICrossAttnCollator(tokenizer=self.tokenizer, max_length=max_length)
        else:
            print("[PPIDataModule] Using ConcatCollator")
            self.collate_fn = PPIConcatCollator(tokenizer=self.tokenizer, max_length=max_length)
        
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

        # =====================================================
        # STEP 1: Split Train/Val First
        # =====================================================
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
            
            groups = pairs_df[split_col].unique()
            np.random.seed(seed)
            np.random.shuffle(groups)
            
            val_count = max(1, int(len(groups) * (1 - train_ratio)))
            val_groups = set(groups[:val_count])
            train_groups = set(groups[val_count:])
            
            train_df = pairs_df[pairs_df[split_col].isin(train_groups)]
            val_df = pairs_df[pairs_df[split_col].isin(val_groups)]
            
            if self.is_main_process:
                print(f"  Train {split_col}s: {len(train_groups):,}")
                print(f"  Val {split_col}s: {len(val_groups):,}")
        else:
            raise ValueError(f"Unknown split_strategy: {strategy}")

        # =====================================================
        # STEP 2: Undersample ONLY the Training Set
        # =====================================================
        undersample = self.cfg.data.get("undersample", False)
        if undersample:
            undersample_ratio = self.cfg.data.get("undersample_ratio", 1.0)
            confidence_threshold = self.cfg.data.get("confidence_threshold", 0.5)
            
            # Split train into binders (positive) and non-binders (negative)
            positive_train = train_df[train_df["confidence"] >= confidence_threshold]
            negative_train = train_df[train_df["confidence"] < confidence_threshold]
            
            n_pos = len(positive_train)
            n_neg = len(negative_train)
            
            if self.is_main_process:
                print(f"[PPIDataModule] Before undersampling (TRAIN ONLY):")
                print(f"  Positive (conf >= {confidence_threshold}): {n_pos:,} ({100*n_pos/len(train_df):.1f}%)")
                print(f"  Negative (conf < {confidence_threshold}): {n_neg:,} ({100*n_neg/len(train_df):.1f}%)")
            
            # Undersample the majority class in training
            n_negative_keep = int(n_pos * undersample_ratio)
            n_negative_keep = min(n_negative_keep, n_neg)
            
            negative_train_sampled = negative_train.sample(
                n=n_negative_keep, 
                random_state=seed
            )
            
            # Recombine and shuffle training set
            train_df = pd.concat([positive_train, negative_train_sampled], ignore_index=True)
            train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
            
            if self.is_main_process:
                print(f"[PPIDataModule] After undersampling (ratio={undersample_ratio}):")
                print(f"  Train Total: {len(train_df):,}")
                print(f"  Train Positive: {len(positive_train):,}")
                print(f"  Train Negative: {n_negative_keep:,}")

        # Final summary print
        if self.is_main_process:
            print(f"[PPIDataModule] Final Split - Train: {len(train_df):,}, Val: {len(val_df):,}")
        
        # =====================================================
        # STEP 3: Create Datasets
        # =====================================================
        balance_clusters = self.cfg.data.get("balance_clusters", False)
        balance_power = self.cfg.data.get("balance_power", 0.5)
        invert_label = self.cfg.data.get("invert_label", True)
        
        self.train_dataset = PPIDataset(
            pairs_df=train_df,
            id2seq=id2seq,
            balance_clusters=balance_clusters,
            balance_power=balance_power,
            invert_label=invert_label,
            verbose=self.is_main_process,
        )
        
        self.val_dataset = PPIDataset(
            pairs_df=val_df,
            id2seq=id2seq,
            balance_clusters=False,
            invert_label=invert_label,
            verbose=self.is_main_process,
        )
        
        # Sampler logic
        if self.train_dataset.weights is not None:
            if self.is_main_process:
                print("[PPIDataModule] Using WeightedRandomSampler")
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