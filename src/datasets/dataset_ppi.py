import os
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import LightningDataModule
from transformers import EsmTokenizer
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig


# ----------------------------------------------------------------------
# PPI-specific Collators (PPI uses 'seq_a'/'seq_b' and 'label' keys,
# different from affinity's 'binder_seq'/'target_seq' and 'log_Aff')
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
# Dataset
# ----------------------------------------------------------------------

class PPIDataset(Dataset):
    """PPI Dataset for confidence regression."""
    def __init__(
        self,
        pairs_df: pd.DataFrame,
        id2seq: Dict[str, str],
        invert_label: bool = True,
        verbose: bool = True,
    ):
        self.id2seq = id2seq
        self.pairs_df = pairs_df.copy()
        self.invert_label = invert_label
        self.verbose = verbose
        
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
        max_length = cfg.model.get("max_length", 1024)
        
        if arch in ["cross_attn", "bi_cross_attn", "binding"]:
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
        # STEP 2: Weighted Sampling for Class Imbalance
        # =====================================================
        # We no longer "undersample" by deleting rows. 
        # Instead, we calculate weights so the model sees classes equally.
        use_weighted_sampler = self.cfg.data.get("use_weighted_sampler", True)
        confidence_threshold = self.cfg.data.get("confidence_threshold", 0.5)
        
        if use_weighted_sampler:
            # 1. Define labels for training set
            train_labels = (train_df["confidence"] >= confidence_threshold).astype(int)
            class_counts = train_labels.value_counts().to_dict()
            
            n_pos = class_counts.get(1, 0)
            n_neg = class_counts.get(0, 0)
            
            if self.is_main_process:
                print(f"[PPIDataModule] Training Class Distribution (Before Weighting):")
                print(f"  Positive (Binders): {n_pos:,} ({100*n_pos/len(train_df):.1f}%)")
                print(f"  Negative (Non-Binders): {n_neg:,} ({100*n_neg/len(train_df):.1f}%)")

            # 2. Calculate inverse frequency weights: 1 / class_count
            # This ensures that in a sample, P(binder) == P(non-binder)
            class_weights = {cls: 1.0 / count for cls, count in class_counts.items() if count > 0}
            sample_weights = train_labels.map(class_weights).values
            
            # 3. Initialize the Sampler
            # num_samples=len(train_df) ensures an "epoch" is still one full pass.
            self.sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_df),
                replacement=True
            )
            
            if self.is_main_process:
                print(f"[PPIDataModule] Initialized WeightedRandomSampler (Label-based)")
        
        # Final summary print
        if self.is_main_process:
            print(f"[PPIDataModule] Final Split - Train: {len(train_df):,}, Val: {len(val_df):,}")

        # =====================================================
        # STEP 3: Create Datasets
        # =====================================================
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