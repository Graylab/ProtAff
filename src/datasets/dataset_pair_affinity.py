import os
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Tuple
from omegaconf import DictConfig

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import LightningDataModule
from transformers import EsmTokenizer

@dataclass
class PairwiseCollator:
    """
    Collates PAIRS of sequences for Siamese training.
    """
    tokenizer: Any
    max_length: int = 1024

    def _tokenize_batch(self, binders: List[str], targets: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper to tokenize a list of Binder+Target strings."""
        eos = self.tokenizer.eos_token
        cls_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        # 1. Raw Strings
        binder_seqs = [str(b).replace(":", eos) for b in binders]
        target_seqs = [str(t).replace(":", eos) for t in targets]

        # 2. Tokenize without padding
        b_encoded = self.tokenizer(binder_seqs, add_special_tokens=False)["input_ids"]
        t_encoded = self.tokenizer(target_seqs, add_special_tokens=False)["input_ids"]

        input_ids_list = []
        mask_list = []

        # 3. Concatenate and Truncate
        for b_ids, t_ids in zip(b_encoded, t_encoded):
            allowed_len = self.max_length - 3 # [CLS] ... [EOS] ... [EOS]
            current_len = len(b_ids) + len(t_ids)

            if current_len > allowed_len:
                excess = current_len - allowed_len
                # Truncate target first
                if len(t_ids) > excess:
                    t_ids = t_ids[:-excess]
                else:
                    rem = excess - len(t_ids)
                    t_ids = []
                    b_ids = b_ids[:-rem]
            
            full_ids = [cls_id] + b_ids + [eos_id] + t_ids + [eos_id]
            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            mask_list.append(torch.ones(len(full_ids), dtype=torch.long))

        # 4. Pad
        batch_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        batch_mask = pad_sequence(mask_list, batch_first=True, padding_value=0)
        
        return batch_ids, batch_mask

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Unpack the list of pairs
        better_binders = [x["better_binder"] for x in batch]
        better_targets = [x["better_target"] for x in batch]
        
        worse_binders = [x["worse_binder"] for x in batch]
        worse_targets = [x["worse_target"] for x in batch]

        b_ids, b_mask = self._tokenize_batch(better_binders, better_targets)
        w_ids, w_mask = self._tokenize_batch(worse_binders, worse_targets)

        # Stack the binary float labels (0.0 or 1.0) into a tensor of shape [Batch_Size]
        better_labels = torch.stack([torch.tensor(x["better_label"], dtype=torch.float) for x in batch])
        worse_labels = torch.stack([torch.tensor(x["worse_label"], dtype=torch.float) for x in batch])

        return {
            "better_input_ids": b_ids,
            "better_mask": b_mask,
            "worse_input_ids": w_ids,
            "worse_mask": w_mask,
            "better_labels": better_labels, 
            "worse_labels": worse_labels
        }

class PairwiseAffinityDataset(Dataset):
    def __init__(self, 
                 base_df: pd.DataFrame, 
                 lookup_csv_path: str, 
                 weight_col: Optional[str] = None, 
                 balance_clusters: bool = False,
                 pairs_per_sample: int = 3,
                 min_margin: float = 0.5,      # Lower Bound (Noise Filter)
                 max_margin: Optional[float] = 6.0): # Upper Bound (Easy Filter)
        """
        Args:
            weight_col: Column to use for Grouping AND Balancing (e.g. 'cluster_id' or 'target_id').
            pairs_per_sample: Number of 'worse' samples to pair with each 'better' sample.
            min_margin: Minimum score difference required.
            max_margin: Maximum score difference allowed (None for no limit).
        """
        # Load Lookup
        lookup_df = pd.read_csv(lookup_csv_path)
        lookup_df['key'] = lookup_df['type'].astype(str) + "_" + lookup_df['id'].astype(str)
        self.id2seq = dict(zip(lookup_df['key'], lookup_df['seq']))
        
        # Prepare Base DF
        self.base_df = base_df.copy()

        # [Dataset] 1. Grouping
        if weight_col and weight_col in self.base_df.columns:
            self.stratify_col = weight_col
        else:
            self.stratify_col = "target_id"
        
        print(f"[Dataset] Grouping strategy: '{self.stratify_col}'")

        # [Dataset] 2. Calculate Weights
        self.cluster_weights = {}
        if balance_clusters:
            # print(f"[Dataset] Calculating balance weights...")
            counts = self.base_df[self.stratify_col].value_counts()
            self.cluster_weights = (1.0 / np.sqrt(counts)).to_dict()

        # [Dataset] 3. Universal Pair Mining (Optimized)
        self.pairs = [] 
        self.weights = [] 
        deltas = [] # For statistics

        print(f"[Dataset] Mining pairs within groups of '{self.stratify_col}'...")
        # print(f"[Dataset] Strategy: Band Mining ({min_margin} <= delta <= {max_margin if max_margin else 'Inf'})")
        
        groups = self.base_df.groupby(self.stratify_col)

        for g_name, group in groups:
            # 1. Sort Data (Best to Worst -> Ascending log_Aff)
            valid_group = group.dropna(subset=["log_Aff"]).sort_values("log_Aff", ascending=True)
            n_total = len(valid_group)
            if n_total < 2: 
                continue

            cur_weight = self.cluster_weights.get(g_name, 1.0)
            
            # 2. Extract values for fast lookup
            aff_values = valid_group["log_Aff"].values 

            # 3. Iterate Top 50% (Anchors)
            n_anchors = max(1, n_total // 2)

            for i in range(n_anchors):
                better_val = aff_values[i]
                
                # --- NEW LOGIC: Band Search ---
                
                # A. Lower Bound (Noise Filter)
                thresh_min = better_val + min_margin
                start_index = np.searchsorted(aff_values, thresh_min, side='right')
                
                # B. Upper Bound (Trivial Filter)
                if max_margin is not None:
                    thresh_max = better_val + max_margin
                    end_index = np.searchsorted(aff_values, thresh_max, side='left')
                else:
                    end_index = n_total
                
                # The valid "Losers" are in the slice [start_index : end_index]
                n_candidates = end_index - start_index
                
                if n_candidates > 0:
                    offsets = np.random.randint(start_index, end_index, size=min(n_candidates, pairs_per_sample))
                    
                    row_better = valid_group.iloc[i]
                    
                    for worse_idx in offsets:
                        row_worse = valid_group.iloc[worse_idx]
                        
                        self.pairs.append((row_better, row_worse))
                        self.weights.append(cur_weight)
                        
                        # Collect Stat
                        deltas.append(aff_values[worse_idx] - better_val)
        
        # [Dataset] 4. Final Stats
        print(f"[Dataset] Generated {len(self.pairs)} pairs.")
        
        kept_groups = len(set(p[0][self.stratify_col] for p in self.pairs))
        total_groups = len(self.base_df[self.stratify_col].unique())
        dropped = total_groups - kept_groups
        
        print(f"[Dataset] Coverage: {kept_groups}/{total_groups} groups used. ({dropped} dropped)")
        
        if len(self.pairs) == 0:
            print(f"WARNING: No valid pairs generated for this split! Check min_margin={min_margin}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row_better, row_worse = self.pairs[idx]
        
        # Construct Lookup Keys
        b_key_better = f"binder_{row_better['binder_id']}"
        t_key_better = f"target_{row_better['target_id']}"
        
        b_key_worse = f"binder_{row_worse['binder_id']}"
        t_key_worse = f"target_{row_worse['target_id']}"

        return {
            "better_binder": self.id2seq.get(b_key_better, ""),
            "better_target": self.id2seq.get(t_key_better, ""),
            "worse_binder": self.id2seq.get(b_key_worse, ""),
            "worse_target": self.id2seq.get(t_key_worse, ""),
            
            # --- Binary Labels for Classification ---
            "better_label": float(row_better['is_binder']),
            "worse_label": float(row_worse['is_binder'])
        }
    
    def get_weight(self, idx):
        return self.weights[idx]

class PairAffinityDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)
        self.num_workers = cfg.data.num_workers if cfg.data.num_workers is not None else os.cpu_count()
        
        self.collate_fn = PairwiseCollator(tokenizer=self.tokenizer, max_length=self.cfg.model.max_length)
        
        self.train_dataset = None
        self.val_dataset = None
        self.sampler = None

    def setup(self, stage: Optional[str] = None):
        print(f"[DataModule] Loading base data from {self.cfg.data.base_csv}")
        base_df = pd.read_csv(self.cfg.data.base_csv)
        
        # --- 1. IDENTIFY SPLIT STRATEGY ---
        # We want to split by 'weight_col' (e.g. cluster_id) to prevent leakage.
        split_col = self.cfg.data.get("weight_col")
        
        if not split_col or split_col not in base_df.columns:
            # Fallback if weight_col is not provided
            split_col = "target_id"
            print(f"[DataModule] 'weight_col' not valid. Splitting by '{split_col}'.")
        else:
            print(f"[DataModule] Stratified Split Strategy: Grouping by '{split_col}'.")

        # --- 2. PERFORM GROUP SPLIT ---
        # Get unique groups (e.g., list of clusters)
        unique_groups = base_df[split_col].unique()
        
        # Shuffle groups for random splitting
        rng = np.random.default_rng(self.cfg.training.seed)
        rng.shuffle(unique_groups)
        
        n_total_groups = len(unique_groups)
        n_val_groups = int(n_total_groups * 0.1) if n_total_groups >= 10 else 1
        n_train_groups = n_total_groups - n_val_groups

        # Assign groups to Train vs Val
        val_groups = set(unique_groups[:n_val_groups])
        train_groups = set(unique_groups[n_val_groups:])
        
        print(f"[DataModule] Splitting {n_total_groups} groups: {n_train_groups} Train Groups, {n_val_groups} Val Groups")

        # Filter the DataFrame based on the groups
        train_df = base_df[base_df[split_col].isin(train_groups)].copy()
        val_df = base_df[base_df[split_col].isin(val_groups)].copy()

        # --- 3. CREATE DATASETS ---
        # Common arguments for both datasets
        ds_kwargs = dict(
            lookup_csv_path=self.cfg.data.lookup_csv, 
            weight_col=self.cfg.data.get("weight_col"), 
            balance_clusters=self.cfg.data.get("balance_clusters", False),
            pairs_per_sample=self.cfg.data.get("pairs_per_sample", 2),
            min_margin=self.cfg.data.get("min_margin", 0.5), 
            max_margin=self.cfg.data.get("max_margin", 6.0)
        )

        print(f"[DataModule] Building TRAIN dataset ({len(train_df)} rows)...")
        self.train_dataset = PairwiseAffinityDataset(base_df=train_df, **ds_kwargs)
        
        print(f"[DataModule] Building VAL dataset ({len(val_df)} rows)...")
        self.val_dataset = PairwiseAffinityDataset(base_df=val_df, **ds_kwargs)

        # --- 4. SETUP SAMPLER (Train Only) ---
        if len(self.train_dataset) > 0 and len(self.train_dataset.weights) > 0:
            print("[DataModule] Setting up WeightedRandomSampler for training...")
            
            # Access weights directly from the dataset
            train_weights = torch.tensor(self.train_dataset.weights, dtype=torch.float)
            
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
    import os
    import pandas as pd
    from torch.utils.data import DataLoader

    print("--- 🧪 RUNNING SELF-TEST FOR DATASET.PY ---")

    # 1. SETUP DUMMY DATA
    dummy_lookup_csv = "temp_lookup_test.csv"
    
    # We create 3 Targets. 
    sequences = [
        {"type": "binder", "id": 101, "seq": "MAAA"}, 
        {"type": "binder", "id": 102, "seq": "MBBB"}, 
        {"type": "binder", "id": 103, "seq": "MCCC"}, 
        {"type": "binder", "id": 201, "seq": "MDDD"}, 
        {"type": "binder", "id": 202, "seq": "MEEE"}, 
        {"type": "binder", "id": 301, "seq": "MFFF"}, 
        {"type": "target", "id": 1, "seq": "TGGG"},
        {"type": "target", "id": 2, "seq": "THHH"},
        {"type": "target", "id": 3, "seq": "TIII"},
    ]
    pd.DataFrame(sequences).to_csv(dummy_lookup_csv, index=False)

    data = [
        # T1 (Group C1)
        {"binder_id": 101, "target_id": 1, "log_Aff": -9.0, "is_binder": 1.0, "cluster_id": "C1"},
        {"binder_id": 102, "target_id": 1, "log_Aff": -6.0, "is_binder": 1.0, "cluster_id": "C1"},
        {"binder_id": 103, "target_id": 1, "log_Aff": 5.0,  "is_binder": 0.0, "cluster_id": "C1"},
        
        # T2 (Group C2)
        {"binder_id": 201, "target_id": 2, "log_Aff": -8.5, "is_binder": 1.0, "cluster_id": "C2"},
        {"binder_id": 202, "target_id": 2, "log_Aff": -8.2, "is_binder": 1.0, "cluster_id": "C2"},
        
        # T3 (Group C3)
        {"binder_id": 301, "target_id": 3, "log_Aff": 5.0,  "is_binder": 0.0, "cluster_id": "C3"}, 
    ]
    df = pd.DataFrame(data)

    try:
        # 3. VERIFY DATASET LOGIC DIRECTLY
        print("\n> Initializing Single Dataset for Logic Check...")
        ds = PairwiseAffinityDataset(
            base_df=df,
            lookup_csv_path=dummy_lookup_csv,
            balance_clusters=True,
            weight_col="cluster_id", # Test Using Cluster ID
            pairs_per_sample=3,
            min_margin=0.5, 
            max_margin=5.0 
        )

        print(f"> Dataset Length: {len(ds)} pairs")
        
        if len(ds) == 0:
            print("❌ FAILURE: Dataset is empty.")
        else:
            print("✅ SUCCESS: Pairs generated.")
            print(f"  Sample 0: {ds[0]['better_binder']} > {ds[0]['worse_binder']}")
            print(f"  Sample 0 Labels: Better={ds[0]['better_label']}, Worse={ds[0]['worse_label']}")

        # 4. TEST COLLATOR
        print("\n> Testing Collator...")
        class MockTokenizer:
            cls_token_id, pad_token_id, eos_token_id = 0, 1, 2
            eos_token = "<eos>"
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [[5, 6, 7] for _ in text]}
        
        collator = PairwiseCollator(tokenizer=MockTokenizer(), max_length=128)
        dl = DataLoader(ds, batch_size=2, collate_fn=collator)
        batch = next(iter(dl))
        
        if batch['better_input_ids'].shape[0] == len(batch['better_input_ids']):
             print("✅ SUCCESS: Batch dimensions correct.")
        
        if 'better_labels' in batch:
             print(f"✅ SUCCESS: Collator returned labels. Shape: {batch['better_labels'].shape}")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if os.path.exists(dummy_lookup_csv):
            os.remove(dummy_lookup_csv)