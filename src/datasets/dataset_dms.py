import os
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Tuple
from omegaconf import DictConfig

from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import LightningDataModule
from transformers import EsmTokenizer

@dataclass
class DMSCollator:
    """
    Collator for DMS Concat Training (Regression Only).
    Format: [CLS] Mutant [EOS] Wildtype [EOS] [PAD]...
    """
    tokenizer: Any
    max_length: int = 2048 

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        eos = self.tokenizer.eos_token
        
        # 1. Prepare Strings
        wt_seqs = [f["wt_seq"] for f in features]
        mut_seqs = [f["mut_seq"] for f in features]
        
        # 2. Tokenize WITHOUT padding/specials first
        wt_encoded = self.tokenizer(wt_seqs, add_special_tokens=False)
        mut_encoded = self.tokenizer(mut_seqs, add_special_tokens=False)
        
        input_ids_list = []
        attention_mask_list = []
        
        cls_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        
        # 3. Concatenate: [CLS] Mutant [EOS] Wildtype [EOS]
        for w_ids, m_ids in zip(wt_encoded["input_ids"], mut_encoded["input_ids"]):
            full_ids = [cls_id] + m_ids + [eos_id] + w_ids + [eos_id]
            
            # Safety Truncation
            if len(full_ids) > self.max_length:
                full_ids = full_ids[:self.max_length]
                if full_ids[-1] != eos_id:
                    full_ids[-1] = eos_id
            
            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            attention_mask_list.append(torch.ones(len(full_ids), dtype=torch.long))

        # 4. Dynamic Padding
        batch_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        batch_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        # 5. Labels (Regression Only)
        reg_labels = [f["reg_label"] for f in features]

        return {
            "input_ids": batch_ids,
            "attention_mask": batch_mask,
            "reg_labels": torch.tensor(reg_labels, dtype=torch.float32).unsqueeze(1),
        }


class DMSDataset(Dataset):
    def __init__(
        self, 
        mutant_csv: str, 
        wt_csv: str, 
        max_length: int = 2048, 
        crop_method: str = "smart", 
        weight_col: Optional[str] = None, 
        balance_clusters: bool = False
    ):
        self.mut_df = pd.read_csv(mutant_csv)
        self.wt_df = pd.read_csv(wt_csv)
        self.crop_method = crop_method
        
        # --- BUDGET LOGIC (Fixed for Concat) ---
        # We ALWAYS need to fit 2 sequences + 3 special tokens.
        # Budget per sequence = (Max - 3) / 2
        self.seq_crop_len = (max_length - 3) // 2
        
        self.wt_lookup = dict(zip(self.wt_df['filename'], self.wt_df['target_seq']))
        self.mut_df = self.mut_df[self.mut_df['filename'].isin(self.wt_lookup)].copy()
        
        # --- Weighting Logic ---
        self.weights = None
        if balance_clusters and weight_col:
            print(f"[DMS] Balancing training data based on cluster: {weight_col}")
            counts = self.mut_df[weight_col].value_counts()
            cluster_sizes = self.mut_df[weight_col].map(counts)
            self.weights = (1.0 / np.sqrt(cluster_sizes)).to_numpy(dtype=np.float32)
        elif weight_col:
            print(f"[DMS] Using raw weights from column: {weight_col}")
            self.weights = pd.to_numeric(self.mut_df[weight_col], errors='coerce').fillna(1.0).to_numpy(dtype=np.float32)

        self.mut_data = self.mut_df[["filename", "mutated_sequence", "DMS_score_normalized"]].to_dict('records')

    def _smart_crop(self, wt: str, mut: str) -> Tuple[str, str]:
        """Centers the crop window on the first mismatch."""
        max_len = self.seq_crop_len
        len_wt, len_mut = len(wt), len(mut)

        if len_wt <= max_len and len_mut <= max_len:
            return wt, mut

        # 1. Find mismatch
        mutation_idx = 0
        found = False
        limit = min(len_wt, len_mut)
        for i in range(limit):
            if wt[i] != mut[i]:
                mutation_idx = i
                found = True
                break
        
        if not found:
            mutation_idx = limit // 2

        # 2. Define Window
        half_win = max_len // 2
        start = max(0, mutation_idx - half_win)
        end = start + max_len

        # 3. Shift window if we hit end of MUTANT (priority)
        if end > len_mut:
            end = len_mut
            start = max(0, end - max_len)

        return wt[start:end], mut[start:end]

    def __len__(self):
        return len(self.mut_data)

    def __getitem__(self, idx):
        row = self.mut_data[idx]
        full_wt = self.wt_lookup[row['filename']]
        full_mut = row['mutated_sequence']

        if self.crop_method == "smart":
            crop_wt, crop_mut = self._smart_crop(full_wt, full_mut)
        else:
            crop_wt = full_wt[:self.seq_crop_len]
            crop_mut = full_mut[:self.seq_crop_len]

        return {
            "wt_seq": crop_wt,
            "mut_seq": crop_mut,
            "reg_label": float(row['DMS_score_normalized']),
        }
    
    def get_weight(self, idx):
        return self.weights[idx] if self.weights is not None else 1.0


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
        mut_path = self.cfg.data.mutant_csv
        wt_path = self.cfg.data.wt_csv
        weight_col = self.cfg.data.get("weight_col", None)
        do_balance = self.cfg.data.get("balance_clusters", False)
        crop_method = self.cfg.data.get("crop_method", "smart")
        
        print(f"[DMS] Cropping Strategy: {crop_method.upper()}")

        full_dataset = DMSDataset(
            mutant_csv=mut_path, 
            wt_csv=wt_path, 
            max_length=self.cfg.model.max_length,
            crop_method=crop_method,
            weight_col=weight_col, 
            balance_clusters=do_balance
        )
        
        total_len = len(full_dataset)
        val_len = 1 if total_len < 10 else int(total_len * 0.1)
        train_len = total_len - val_len

        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_len, val_len],
            generator=torch.Generator().manual_seed(self.cfg.training.seed)
        )

        if full_dataset.weights is not None:
            train_weights = torch.tensor(
                [full_dataset.get_weight(i) for i in self.train_dataset.indices], 
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