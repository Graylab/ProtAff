import os
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from omegaconf import DictConfig

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pytorch_lightning import LightningDataModule
from transformers import EsmTokenizer

# =========================================================================
# 1. Single-Sequence Collator (Regression)
# =========================================================================
@dataclass
class DMSCollator:
    """
    Collator for Single-Sequence DMS Regression.
    Input: [CLS] Mutant [EOS]
    Output: input_ids, attention_mask, reg_labels
    """
    tokenizer: Any
    max_length: int = 1024 

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. Prepare Strings (MUTANT ONLY)
        mut_seqs = [f["mut_seq"] for f in features]
        
        # 2. Tokenize
        encoded = self.tokenizer(
            mut_seqs, 
            add_special_tokens=True, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        # 3. Stack Labels
        reg_labels = [f["reg_label"] for f in features]

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "reg_labels": torch.tensor(reg_labels, dtype=torch.float32).unsqueeze(1), # [Batch, 1]
        }

# =========================================================================
# 2. DMS Dataset
# =========================================================================
class DMSDataset(Dataset):
    def __init__(
        self, 
        dataframe: pd.DataFrame, # Changed: Accept DataFrame directly
        wt_csv: str, 
        max_length: int = 1024, 
        crop_method: str = "smart",
        weight_col: Optional[str] = None,
        balance_clusters: bool = False,
        normalize_labels: bool = True, # New: Auto-normalization
        invert_score: bool = True     # New: Optional sign flip
    ):
        self.mut_df = dataframe.copy() # Work on a copy
        self.wt_df = pd.read_csv(wt_csv)
        self.crop_method = crop_method
        self.invert_score = invert_score
        
        # Budget for Single Sequence (Standard ESM length)
        self.seq_crop_len = max_length - 2 # -2 for [CLS], [EOS]
        
        # Create Lookup for WT sequences
        self.wt_lookup = dict(zip(self.wt_df['filename'], self.wt_df['target_seq']))
        
        # Filter mutants to ensure we have their WT (Safety check)
        self.mut_df = self.mut_df[self.mut_df['filename'].isin(self.wt_lookup)].copy()

        # --- Label Normalization (Critical for Regression) ---
        self.label_mean = 0.0
        self.label_std = 1.0
        
        # Extract raw scores
        raw_scores = pd.to_numeric(self.mut_df["DMS_score_normalized"], errors='coerce').values
        
        if invert_score:
            print("[Dataset] ⚠️ Inverting scores (multiplying by -1.0).")
            raw_scores = -1.0 * raw_scores
            
        if normalize_labels:
            self.label_mean = np.nanmean(raw_scores)
            self.label_std = np.nanstd(raw_scores) + 1e-8
            print(f"[Dataset] Normalizing Labels -> Mean: {self.label_mean:.4f}, Std: {self.label_std:.4f}")
        else:
            print("[Dataset] Using RAW labels (No Z-Score Normalization).")

        # Store mean/std for use in __getitem__
        self.norm_params = (self.label_mean, self.label_std)

        # --- Weighting Logic ---
        self.weights = None
        if balance_clusters and weight_col:
            print(f"[Dataset] Balancing training data based on cluster: {weight_col}")
            counts = self.mut_df[weight_col].value_counts()
            cluster_sizes = self.mut_df[weight_col].map(counts)
            self.weights = (1.0 / np.sqrt(cluster_sizes)).to_numpy(dtype=np.float32)
        elif weight_col:
            print(f"[Dataset] Using raw weights from column: {weight_col}")
            self.weights = pd.to_numeric(self.mut_df[weight_col], errors='coerce').fillna(1.0).to_numpy(dtype=np.float32)

        self.mut_data = self.mut_df[["filename", "mutated_sequence", "DMS_score_normalized"]].to_dict('records')

    def _smart_crop(self, wt: str, mut: str) -> str:
        """
        Centers the crop window on the mutation site.
        """
        max_len = self.seq_crop_len
        len_mut = len(mut)

        if len_mut <= max_len:
            return mut

        # 1. Find mismatch
        limit = min(len(wt), len_mut)
        mutation_idx = limit // 2 
        
        for i in range(limit):
            if wt[i] != mut[i]:
                mutation_idx = i
                break
        
        # 2. Define Window
        half_win = max_len // 2
        start = max(0, mutation_idx - half_win)
        end = start + max_len

        # 3. Shift window if we hit end
        if end > len_mut:
            end = len_mut
            start = max(0, end - max_len)

        return mut[start:end]

    def __len__(self):
        return len(self.mut_data)

    def __getitem__(self, idx):
        row = self.mut_data[idx]
        filename = row['filename']
        full_wt = self.wt_lookup[filename]
        full_mut = row['mutated_sequence']
        
        # Crop
        if self.crop_method == "smart":
            crop_mut = self._smart_crop(full_wt, full_mut)
        else:
            crop_mut = full_mut[:self.seq_crop_len]

        # Label Processing
        raw_val = float(row['DMS_score_normalized'])
        
        if self.invert_score:
            raw_val = -1.0 * raw_val
            
        # Z-Score Normalization
        # (val - mean) / std
        norm_val = (raw_val - self.norm_params[0]) / self.norm_params[1]

        return {
            "mut_seq": crop_mut,
            "reg_label": norm_val, 
        }

    def get_weight(self, idx):
        return self.weights[idx] if self.weights is not None else 1.0

# =========================================================================
# 3. Data Module
# =========================================================================
class DMSDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)
        self.num_workers = cfg.data.num_workers if cfg.data.num_workers is not None else os.cpu_count()
        
        self.collate_fn = DMSCollator(tokenizer=self.tokenizer, max_length=self.cfg.model.max_length)
        
        self.train_dataset = None
        self.val_dataset = None
        self.sampler = None

    def setup(self, stage: Optional[str] = None):
        print(f"[DMS] Loading Data. Crop Strategy: {self.cfg.data.get('crop_method', 'smart')}")

        # 1. Load Full Dataframe
        full_df = pd.read_csv(self.cfg.data.mutant_csv)
        
        # 2. STRICT TARGET SPLIT (Unseen Targets in Validation)
        # We split by 'filename' (target ID), NOT by row.
        all_targets = full_df['filename'].unique()
        
        # Shuffle targets deterministically
        rng = np.random.default_rng(self.cfg.training.seed)
        rng.shuffle(all_targets)
        
        # Define Split
        val_count = max(1, int(len(all_targets) * 0.1)) # 10% validation
        val_targets = set(all_targets[:val_count])
        train_targets = set(all_targets[val_count:])
        
        print(f"[Split] Total Targets: {len(all_targets)}")
        print(f"[Split] Train Targets: {len(train_targets)} | Val Targets: {len(val_targets)} (STRICTLY UNSEEN)")
        
        # Create Sub-DataFrames
        train_df = full_df[full_df['filename'].isin(train_targets)].copy()
        val_df = full_df[full_df['filename'].isin(val_targets)].copy()

        # 3. Initialize Datasets (Passing DataFrames)
        # Note: We can invert score if config requests it, but default is False (Higher is Better)
        invert = self.cfg.data.get("invert_score", False)
        normalize = self.cfg.data.get("normalize_labels", True)

        self.train_dataset = DMSDataset(
            dataframe=train_df, 
            wt_csv=self.cfg.data.wt_csv, 
            max_length=self.cfg.model.max_length,
            crop_method=self.cfg.data.get("crop_method", "smart"),
            weight_col=self.cfg.data.get("weight_col", None),
            balance_clusters=self.cfg.data.get("balance_clusters", False),
            normalize_labels=normalize,
            invert_score=invert
        )

        # For Validation, we ideally use Train stats to normalize, but self-normalization is fine for monitoring
        self.val_dataset = DMSDataset(
            dataframe=val_df, 
            wt_csv=self.cfg.data.wt_csv, 
            max_length=self.cfg.model.max_length,
            crop_method=self.cfg.data.get("crop_method", "smart"),
            weight_col=None, # No weighting for validation
            balance_clusters=False,
            normalize_labels=normalize,
            invert_score=invert
        )
        
        # 4. Sampler for Training
        if self.train_dataset.weights is not None:
            print("[DMS] Setting up WeightedRandomSampler for training...")
            # We must iterate via indices because dataset is now a subset
            train_weights = torch.tensor(
                [self.train_dataset.get_weight(i) for i in range(len(self.train_dataset))], 
                dtype=torch.float
            )
            self.sampler = WeightedRandomSampler(
                weights=train_weights, 
                num_samples=len(train_weights), 
                replacement=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.cfg.training.batch_size, 
            collate_fn=self.collate_fn, 
            num_workers=self.num_workers,
            sampler=self.sampler,
            shuffle=(self.sampler is None),
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.cfg.training.batch_size, 
            collate_fn=self.collate_fn, 
            num_workers=self.num_workers, 
            pin_memory=True
        )

if __name__ == '__main__':
    print("--- 🧪 RUNNING SELF-TEST FOR DATASET.PY (Regression Mode) ---")
    
    # 1. SETUP DUMMY DATA
    dummy_wt_csv = "temp_wt.csv"
    dummy_mut_csv = "temp_mut.csv"
    
    # Protein A and B (WT)
    wt_data = [
        {"filename": "protein_A", "target_seq": "MKTIIALSYIFCLVFADYKDDDDK"}, 
        {"filename": "protein_B", "target_seq": "MSKGEELFTGVVPILVELDGDVNG"}, 
    ]
    pd.DataFrame(wt_data).to_csv(dummy_wt_csv, index=False)
    
    # Mutants (Higher is better, e.g., Fitness)
    mut_data = [
        # Protein A variants
        {"filename": "protein_A", "mutated_sequence": "MKTIIALSYIFCLVFADYKDDDDE", "DMS_score_normalized": 5.0, "cluster_id": "C1"},
        {"filename": "protein_A", "mutated_sequence": "MKTIIALSYIFCLVFADYKDDDDA", "DMS_score_normalized": 1.0, "cluster_id": "C1"},
        # Protein B variants
        {"filename": "protein_B", "mutated_sequence": "MSKGEELFTGVVPILVELDGDVNA", "DMS_score_normalized": 10.0, "cluster_id": "C2"},
        {"filename": "protein_B", "mutated_sequence": "MSKGEELFTGVVPILVELDGDVNK", "DMS_score_normalized": 2.0, "cluster_id": "C2"},
    ]
    pd.DataFrame(mut_data).to_csv(dummy_mut_csv, index=False)

    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            "model": {"name": "facebook/esm2_t6_8M_UR50D", "max_length": 128},
            "data": {
                "mutant_csv": dummy_mut_csv,
                "wt_csv": dummy_wt_csv,
                "num_workers": 0, 
                "crop_method": "smart",
                "weight_col": "cluster_id", 
                "balance_clusters": True,
                "invert_score": False, # Should keep positive scores positive
                "normalize_labels": True
            },
            "training": {"batch_size": 2, "seed": 42}
        })

        # 3. INITIALIZE DATAMODULE
        print("\n> Initializing DataModule...")
        dm = DMSDataModule(cfg)
        dm.setup()
        
        # 4. CHECK BATCH
        print("\n> Fetching a batch...")
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        
        labels = batch['reg_labels']
        print(f"  Labels (Normalized): {labels.squeeze().tolist()}")
        
        # Verify shape
        if labels.shape[1] == 1:
            print("✅ SUCCESS: Labels are [Batch, 1].")
        else:
            print(f"❌ FAILURE: Labels shape {labels.shape}")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if os.path.exists(dummy_wt_csv): os.remove(dummy_wt_csv)
        if os.path.exists(dummy_mut_csv): os.remove(dummy_mut_csv)