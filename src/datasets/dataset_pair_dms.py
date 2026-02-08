import os
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Tuple
from omegaconf import DictConfig

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pytorch_lightning import LightningDataModule
from transformers import EsmTokenizer

from src.datasets.collators import tokenize_cross_attn


# =========================================================================
# 1. Pairwise Collator for Cross-Attention (Mutant=Binder, WT=Target)
# =========================================================================
@dataclass
class PairDMSCollator:
    """
    Collator for DMS pairwise ranking with cross-attention architecture.
    Output keys match PairAffinityModule format exactly.
    
    Mapping:
    - better_binder = better fitness mutant
    - better_target = wildtype
    - worse_binder = worse fitness mutant  
    - worse_target = wildtype
    """
    tokenizer: Any
    arch: str = "cross_attn"
    max_length: int = 1024

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        better_mut = [x["better_mutant"] for x in batch]
        better_wt = [x["better_wildtype"] for x in batch]
        worse_mut = [x["worse_mutant"] for x in batch]
        worse_wt = [x["worse_wildtype"] for x in batch]
        deltas = torch.tensor([x["delta"] for x in batch], dtype=torch.float32)

        if self.arch in ["cross_attn", "bi_cross_attn", "binding"]:
            better_enc = tokenize_cross_attn(self.tokenizer, better_mut, better_wt, self.max_length)
            worse_enc = tokenize_cross_attn(self.tokenizer, worse_mut, worse_wt, self.max_length)

            return {
                "better_binder_ids": better_enc["binder_ids"],
                "better_binder_mask": better_enc["binder_mask"],
                "better_target_ids": better_enc["target_ids"],
                "better_target_mask": better_enc["target_mask"],
                "worse_binder_ids": worse_enc["binder_ids"],
                "worse_binder_mask": worse_enc["binder_mask"],
                "worse_target_ids": worse_enc["target_ids"],
                "worse_target_mask": worse_enc["target_mask"],
                "delta": deltas,
            }
        else:
            # Concat architecture - combine mutant + wt into single sequence
            better_combined = [f"{m}:{w}" for m, w in zip(better_mut, better_wt)]
            worse_combined = [f"{m}:{w}" for m, w in zip(worse_mut, worse_wt)]
            
            better_enc = self.tokenizer(
                better_combined, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt"
            )
            worse_enc = self.tokenizer(
                worse_combined, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt"
            )
            
            return {
                "better_input_ids": better_enc["input_ids"],
                "better_mask": better_enc["attention_mask"],
                "worse_input_ids": worse_enc["input_ids"],
                "worse_mask": worse_enc["attention_mask"],
                "delta": deltas
            }


# =========================================================================
# 2. DMS Pairwise Dataset
# =========================================================================
class PairDMSDataset(Dataset):
    """
    DMS dataset for pairwise ranking pre-training.
    Creates pairs: (better_mutant, worse_mutant) relative to same wildtype.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        wt_csv: str,
        pairs_per_anchor: int = 3,
        min_margin: float = 0.3,
        max_pairs_per_protein: int = 2000,
        max_length: int = 1024,
        crop_method: str = "smart",
        invert_score: bool = False,
        verbose: bool = True
    ):
        self.verbose = verbose
        self.crop_method = crop_method
        self.seq_crop_len = max_length - 2  # -2 for [CLS], [EOS]
        
        self._log(f"\n{'='*25} DMS PAIRWISE DATASET {'='*25}")
        
        # Load wildtype sequences
        wt_df = pd.read_csv(wt_csv)
        self.wt_lookup = dict(zip(wt_df['filename'], wt_df['target_seq']))
        
        # Filter to proteins with known WT
        mut_df = dataframe[dataframe['filename'].isin(self.wt_lookup)].copy()
        
        self._log(f"[*] Raw DMS samples: {len(mut_df):,}")
        self._log(f"[*] Unique proteins: {mut_df['filename'].nunique():,}")
        
        # Invert scores if needed (so higher = better)
        if invert_score:
            self._log(f"[*] Inverting scores (× -1)")
            mut_df['fitness'] = -1.0 * mut_df['DMS_score_normalized']
        else:
            mut_df['fitness'] = mut_df['DMS_score_normalized']
        
        # Build pairs
        self.better_mut, self.worse_mut = [], []
        self.better_wt, self.worse_wt = [], []
        self.deltas = []
        self.filenames = []
        
        for filename, group in mut_df.groupby('filename'):
            wt_seq = self.wt_lookup[filename]
            group = group.sort_values('fitness')
            
            mutants = group['mutated_sequence'].values
            scores = group['fitness'].values
            n = len(mutants)
            
            if n < 2:
                continue
            
            # Find valid pairs with margin
            pairs_added = 0
            starts = np.searchsorted(scores, scores + min_margin, side='right')
            valid_anchors = np.where(starts < n)[0]
            
            if len(valid_anchors) > max_pairs_per_protein:
                valid_anchors = np.random.choice(valid_anchors, max_pairs_per_protein, replace=False)
            
            for a_idx in valid_anchors:
                if pairs_added >= max_pairs_per_protein:
                    break
                
                possible = list(range(starts[a_idx], n))
                if not possible:
                    continue
                
                chosen = np.random.choice(possible, min(len(possible), pairs_per_anchor), replace=False)
                
                for b_idx in chosen:
                    # Higher fitness = better
                    # Crop sequences
                    better_mut_crop = self._smart_crop(wt_seq, mutants[b_idx])
                    worse_mut_crop = self._smart_crop(wt_seq, mutants[a_idx])
                    wt_crop = self._crop_wt(wt_seq)
                    
                    self.better_mut.append(better_mut_crop)
                    self.better_wt.append(wt_crop)
                    self.worse_mut.append(worse_mut_crop)
                    self.worse_wt.append(wt_crop)
                    self.deltas.append(scores[b_idx] - scores[a_idx])
                    self.filenames.append(filename)
                    pairs_added += 1
        
        self._log(f"[*] Total DMS pairs: {len(self.better_mut):,}")
        self._log(f"{'='*60}\n")
    
    def _smart_crop(self, wt: str, mut: str) -> str:
        """Centers the crop window on the mutation site."""
        max_len = self.seq_crop_len
        
        if len(mut) <= max_len:
            return mut
        
        # Find first mismatch
        limit = min(len(wt), len(mut))
        mutation_idx = limit // 2
        
        for i in range(limit):
            if wt[i] != mut[i]:
                mutation_idx = i
                break
        
        # Define window
        half_win = max_len // 2
        start = max(0, mutation_idx - half_win)
        end = start + max_len
        
        if end > len(mut):
            end = len(mut)
            start = max(0, end - max_len)
        
        return mut[start:end]
    
    def _crop_wt(self, wt: str) -> str:
        """Crop wildtype sequence."""
        if len(wt) <= self.seq_crop_len:
            return wt
        # Center crop for WT
        start = (len(wt) - self.seq_crop_len) // 2
        return wt[start:start + self.seq_crop_len]
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def __len__(self):
        return len(self.better_mut)
    
    def __getitem__(self, idx):
        return {
            "better_mutant": self.better_mut[idx],
            "better_wildtype": self.better_wt[idx],
            "worse_mutant": self.worse_mut[idx],
            "worse_wildtype": self.worse_wt[idx],
            "delta": self.deltas[idx]
        }


# =========================================================================
# 3. DMS Pairwise DataModule
# =========================================================================
class PairDMSDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)
        self.num_workers = cfg.data.get("num_workers", 4)
        
        self.collate_fn = PairDMSCollator(
            tokenizer=self.tokenizer,
            arch=cfg.model.get("arch", "cross_attn"),
            max_length=cfg.model.max_length
        )
        
        self.train_dataset = None
        self.val_dataset = None
        self.sampler = None

    @property
    def is_main_process(self) -> bool:
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        return rank == 0

    def setup(self, stage: Optional[str] = None):
        if self.is_main_process:
            print(f"\n[DMS Pairwise] Loading Data...")
            print(f"[DMS Pairwise] Architecture: {self.cfg.model.get('arch', 'cross_attn')}")
        
        full_df = pd.read_csv(self.cfg.data.mutant_csv)
        
        seed = self.cfg.training.get("seed", 42)
        train_ratio = self.cfg.training.get("train_val_split", 0.9)
        strategy = self.cfg.training.get("split_strategy", "group")
        
        if self.is_main_process:
            print(f"[DMS Pairwise] Split strategy: {strategy}")
        
        if strategy == "random":
            # Random split (rows, not proteins)
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(
                full_df,
                train_size=train_ratio,
                random_state=seed,
                shuffle=True
            )
            if self.is_main_process:
                print(f"[Split] Random split")
                print(f"[Split] Train samples: {len(train_df)} | Val samples: {len(val_df)}")
        
        elif strategy == "group":
            # Group split by filename (protein) - val has UNSEEN proteins
            train_df, val_df = self._group_split(full_df, train_ratio, seed)
        
        else:
            raise ValueError(f"Unknown split_strategy: {strategy}. Options: ['random', 'group']")
        
        # Create datasets
        self.train_dataset = PairDMSDataset(
            dataframe=train_df,
            wt_csv=self.cfg.data.wt_csv,
            pairs_per_anchor=self.cfg.data.get("pairs_per_anchor", 3),
            min_margin=self.cfg.data.get("min_margin", 0.3),
            max_pairs_per_protein=self.cfg.data.get("max_pairs_per_protein", 2000),
            max_length=self.cfg.model.max_length,
            crop_method=self.cfg.data.get("crop_method", "smart"),
            invert_score=self.cfg.data.get("invert_score", False),
            verbose=self.is_main_process
        )
        
        self.val_dataset = PairDMSDataset(
            dataframe=val_df,
            wt_csv=self.cfg.data.wt_csv,
            pairs_per_anchor=2,
            min_margin=self.cfg.data.get("min_margin", 0.3),
            max_pairs_per_protein=500,
            max_length=self.cfg.model.max_length,
            crop_method=self.cfg.data.get("crop_method", "smart"),
            invert_score=self.cfg.data.get("invert_score", False),
            verbose=self.is_main_process
        )
        
        # Setup weighted sampler
        self._setup_sampler()

    def _group_split(self, full_df: pd.DataFrame, train_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split by filename (protein) - validation has completely unseen proteins."""
        all_proteins = full_df['filename'].unique()
        
        # Count samples per protein
        counts = full_df['filename'].value_counts()
        singleton_proteins = counts[counts == 1].index.tolist()
        multi_sample_proteins = counts[counts > 1].index.tolist()
        
        if self.is_main_process:
            print(f"[Split] Total proteins: {len(all_proteins)}")
            print(f"[Split] Singleton proteins (1 sample): {len(singleton_proteins)}")
            print(f"[Split] Multi-sample proteins (>1 sample): {len(multi_sample_proteins)}")
        
        # Shuffle and split multi-sample proteins
        np.random.seed(seed)
        np.random.shuffle(multi_sample_proteins)
        
        val_count = max(1, int(len(multi_sample_proteins) * (1 - train_ratio)))
        val_proteins = set(multi_sample_proteins[:val_count])
        train_proteins_multi = set(multi_sample_proteins[val_count:])
        
        # Add singletons to train (can't create pairs from singletons anyway)
        train_proteins = train_proteins_multi.union(set(singleton_proteins))
        
        # Create splits
        train_df = full_df[full_df['filename'].isin(train_proteins)].copy()
        val_df = full_df[full_df['filename'].isin(val_proteins)].copy()
        
        # Verify no overlap
        overlap = train_proteins & val_proteins
        if len(overlap) > 0:
            raise ValueError(f"ERROR: Train/Val protein overlap: {overlap}")
        
        if self.is_main_process:
            print(f"[Split] ✓ Train proteins: {len(train_proteins)}")
            print(f"[Split]   - Multi-sample: {len(train_proteins_multi)}")
            print(f"[Split]   - Singletons: {len(singleton_proteins)}")
            print(f"[Split] ✓ Val proteins: {len(val_proteins)} (COMPLETELY UNSEEN)")
            print(f"[Split] ✓ Overlap check: {len(overlap)} (MUST be 0)")
            print(f"[Split] Train samples: {len(train_df)} | Val samples: {len(val_df)}")
        
        return train_df, val_df

    def _setup_sampler(self):
        """Weighted sampling by filename (protein) frequency."""
        balance_clusters = self.cfg.data.get("balance_clusters", True)
        balance_power = self.cfg.data.get("balance_power", 0.5)
        
        if not balance_clusters:
            self.sampler = None
            if self.is_main_process:
                print(f"\n[Sampler] No balancing (shuffle only)")
            return
        
        if self.is_main_process:
            print(f"\n[Sampler] Balancing by protein (filename)")
            print(f"[Sampler] Balance power: {balance_power}")
        
        # Count pairs per protein
        protein_counts = pd.Series(self.train_dataset.filenames).value_counts()
        
        if self.is_main_process:
            print(f"[Sampler] Top proteins by pair count:")
            print(protein_counts.head(5).to_string())
        
        # Compute weights: 1 / count^power
        counts_dict = protein_counts.to_dict()
        pair_weights = []
        
        for filename in self.train_dataset.filenames:
            freq = counts_dict[filename]
            weight = 1.0 / np.power(freq, balance_power)
            pair_weights.append(weight)
        
        pair_weights = np.array(pair_weights, dtype=np.float32)
        
        if self.is_main_process:
            print(f"[Sampler] Weight range: {pair_weights.min():.4f} to {pair_weights.max():.4f}")
            print(f"[Sampler] Weight ratio: {pair_weights.max()/pair_weights.min():.1f}x")
        
        self.sampler = WeightedRandomSampler(
            weights=pair_weights,
            num_samples=len(pair_weights),
            replacement=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True
        )