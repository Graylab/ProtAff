import os
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Tuple
from omegaconf import DictConfig

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
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

        return {
            "better_input_ids": b_ids,
            "better_mask": b_mask,
            "worse_input_ids": w_ids,
            "worse_mask": w_mask
        }

class PairwiseAffinityDataset(Dataset):
    def __init__(self, 
                 base_df: pd.DataFrame, 
                 lookup_csv_path: str, 
                 weight_col: Optional[str] = None, 
                 balance_clusters: bool = False,
                 pairs_per_sample: int = 3,
                 margin_threshold: float = 0.5):
        """
        Args:
            weight_col: Column to use for Grouping AND Balancing (e.g. 'cluster_id' or 'target_id').
            pairs_per_sample: Number of 'worse' samples to pair with each 'better' sample.
        """
        # 1. Load Lookup
        lookup_df = pd.read_csv(lookup_csv_path)
        lookup_df['key'] = lookup_df['type'].astype(str) + "_" + lookup_df['id'].astype(str)
        self.id2seq = dict(zip(lookup_df['key'], lookup_df['seq']))
        
        # 2. Prepare Base DF
        self.base_df = base_df.copy()
        
        if "is_binder" not in self.base_df.columns:
            print("[Dataset] 'is_binder' missing. Inferring from log_Aff < 4.0...")
            self.base_df["is_binder"] = np.where(self.base_df["log_Aff"] < 4.0, 1.0, 0.0)

        # --- UPDATED LOGIC START ---
        # Determine the primary grouping column (Stratify Key)
        # If weight_col is provided (e.g. 'cluster_id'), use it. Otherwise default to 'target_id'.
        if weight_col and weight_col in self.base_df.columns:
            self.stratify_col = weight_col
        else:
            self.stratify_col = "target_id"
        
        print(f"[Dataset] Grouping and Balancing strategy based on: '{self.stratify_col}'")
        # --- UPDATED LOGIC END ---

        # 3. Calculate Cluster Weights (for sampling)
        self.cluster_weights = {}
        if balance_clusters:
            print(f"[Dataset] Calculating balance weights...")
            counts = self.base_df[self.stratify_col].value_counts()
            
            # Inverse square root weighting to flatten the distribution
            unique_weights = (1.0 / np.sqrt(counts)).to_dict() 
            self.cluster_weights = unique_weights

        # 4. Mine Pairs
        self.pairs = [] # List of (row_better, row_worse, weight)
        self.weights = [] 

        # Group by the stratify_col to ensure we don't cross-compare incompatible groups
        print(f"[Dataset] Mining pairs within groups of '{self.stratify_col}'...")
        groups = self.base_df.groupby(self.stratify_col)

        for g_name, group in groups:
            binders = group[group["is_binder"] == 1]
            decoys = group[group["is_binder"] == 0]

            # Get weight for this group
            if g_name in self.cluster_weights:
                cur_weight = self.cluster_weights[g_name]
            else:
                cur_weight = 1.0

            # Strategy A: Binder > Decoy
            if len(decoys) > 0 and len(binders) > 0:
                for idx, row_better in binders.iterrows():
                    chosen_decoys = decoys.sample(n=min(len(decoys), pairs_per_sample), replace=True)
                    for _, row_worse in chosen_decoys.iterrows():
                        self.pairs.append((row_better, row_worse))
                        self.weights.append(cur_weight)

            # Strategy B: Strong Binder > Weak Binder
            if len(binders) > 1:
                # Sort descending (Higher log_Aff is better)
                sorted_binders = binders.sort_values("log_Aff", ascending=False)
                
                n = len(sorted_binders)
                top = sorted_binders.iloc[:n//2]
                bot = sorted_binders.iloc[n//2:]

                if len(bot) > 0:
                    for idx, row_better in top.iterrows():
                        candidates = bot[bot["log_Aff"] < (row_better["log_Aff"] - margin_threshold)]
                        if len(candidates) > 0:
                            chosen = candidates.sample(n=min(len(candidates), pairs_per_sample))
                            for _, row_worse in chosen.iterrows():
                                self.pairs.append((row_better, row_worse))
                                self.weights.append(cur_weight)

        print(f"[Dataset] Generated {len(self.pairs)} pairs.")
        
        # Coverage Check
        # If grouping by cluster_id, count unique cluster_ids in the pairs
        kept_groups = len(set(p[0][self.stratify_col] for p in self.pairs)) 
        total_groups = len(self.base_df[self.stratify_col].unique())
        dropped = total_groups - kept_groups
        
        print(f"[Dataset] Coverage: {kept_groups}/{total_groups} groups used. ({dropped} dropped)")
        
        if len(self.pairs) == 0:
            raise ValueError("CRITICAL: No valid pairs could be generated!")

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
        
        # 1. Create Pairwise Dataset
        full_dataset = PairwiseAffinityDataset(
            base_df, 
            self.cfg.data.lookup_csv, 
            weight_col=self.cfg.data.get("weight_col"), 
            balance_clusters=self.cfg.data.get("balance_clusters", False),
            pairs_per_sample=self.cfg.data.get("pairs_per_sample", 2),
            margin_threshold=self.cfg.data.get("margin_threshold", 0.5)
        )
        
        # 2. Random Split
        total_len = len(full_dataset)
        val_len = int(total_len * 0.1) if total_len >= 10 else 1
        train_len = total_len - val_len

        print(f"[DataModule] Performing Random Split on PAIRS: {train_len} Train, {val_len} Val")
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_len, val_len],
            generator=torch.Generator().manual_seed(self.cfg.training.seed)
        )

        # 3. Setup Weighted Sampler
        if len(full_dataset.weights) > 0:
            print("[DataModule] Setting up WeightedRandomSampler for training...")
            
            # Extract weights for the specific indices chosen by random_split
            train_indices = self.train_dataset.indices
            train_weights = torch.tensor(
                [full_dataset.get_weight(i) for i in train_indices], 
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
    import os
    import pandas as pd
    from torch.utils.data import DataLoader

    print("--- 🧪 RUNNING SELF-TEST FOR DATASET.PY ---")

    # 1. SETUP DUMMY DATA
    # We need a temporary lookup CSV because the class reads from a path
    dummy_lookup_csv = "temp_lookup_test.csv"
    
    # 3 Clusters/Targets:
    # Target 1: Has Strong (9.0), Weak (6.0), and Decoy (0.0). Should generate pairs.
    # Target 2: Has two Binders (8.5, 8.2). Should pair if margin is small.
    # Target 3: Only Decoys. Should be dropped (Safety check).
    sequences = [
        {"type": "binder", "id": 101, "seq": "MAAA"}, # Strong T1
        {"type": "binder", "id": 102, "seq": "MBBB"}, # Weak T1
        {"type": "binder", "id": 103, "seq": "MCCC"}, # Decoy T1
        {"type": "binder", "id": 201, "seq": "MDDD"}, # Binder T2
        {"type": "binder", "id": 202, "seq": "MEEE"}, # Binder T2
        {"type": "binder", "id": 301, "seq": "MFFF"}, # Decoy T3 (Lonely)
        {"type": "target", "id": 1, "seq": "TGGG"},
        {"type": "target", "id": 2, "seq": "THHH"},
        {"type": "target", "id": 3, "seq": "TIII"},
    ]
    pd.DataFrame(sequences).to_csv(dummy_lookup_csv, index=False)

    # 2. CREATE DATAFRAME
    data = [
        {"binder_id": 101, "target_id": 1, "cluster_id": "c1", "log_Aff": 9.0, "is_binder": 1},
        {"binder_id": 102, "target_id": 1, "cluster_id": "c1", "log_Aff": 6.0, "is_binder": 1},
        {"binder_id": 103, "target_id": 1, "cluster_id": "c1", "log_Aff": 0.0, "is_binder": 0},
        {"binder_id": 201, "target_id": 2, "cluster_id": "c2", "log_Aff": 8.5, "is_binder": 1},
        {"binder_id": 202, "target_id": 2, "cluster_id": "c2", "log_Aff": 8.2, "is_binder": 1},
        {"binder_id": 301, "target_id": 3, "cluster_id": "c3", "log_Aff": 0.0, "is_binder": 0},
    ]
    df = pd.DataFrame(data)

    try:
        # 3. INITIALIZE DATASET
        print("\n> Initializing Dataset...")
        ds = PairwiseAffinityDataset(
            base_df=df,
            lookup_csv_path=dummy_lookup_csv,
            balance_clusters=True,
            pairs_per_sample=2,
            margin_threshold=0.2  # Low threshold to catch the 8.5 vs 8.2 pair
        )

        # 4. VERIFY LOGIC
        print(f"> Dataset Length: {len(ds)} pairs")
        
        # We expect:
        # T1: 101>103, 102>103 (Binder>Decoy) AND 101>102 (Strong>Weak, diff=3.0)
        # T2: 201>202 (Strong>Weak, diff=0.3)
        # T3: Dropped (0 pairs)
        if len(ds) == 0:
            print("❌ FAILURE: Dataset is empty.")
        else:
            print("✅ SUCCESS: Pairs generated.")
            print(f"  Sample 0: {ds[0]['better_binder']} > {ds[0]['worse_binder']}")

        # 5. TEST COLLATOR & SHAPES
        print("\n> Testing Collator...")
        
        # MOCK Tokenizer to avoid downloading ESM weights for this simple test
        class MockTokenizer:
            cls_token_id = 0
            pad_token_id = 1
            eos_token_id = 2
            eos_token = "<eos>"
            def __call__(self, text, add_special_tokens=False):
                # Returns dummy IDs for any string
                return {"input_ids": [[5, 6, 7] for _ in text]}
        
        collator = PairwiseCollator(tokenizer=MockTokenizer(), max_length=128)
        dl = DataLoader(ds, batch_size=2, collate_fn=collator)
        
        batch = next(iter(dl))
        print(f"  Batch keys: {list(batch.keys())}")
        print(f"  Better Input Shape: {batch['better_input_ids'].shape}")
        
        if batch['better_input_ids'].shape[0] == 2:
            print("✅ SUCCESS: Batch dimensions correct.")
        else:
            print("❌ FAILURE: Batch dimension mismatch.")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 6. CLEANUP
        if os.path.exists(dummy_lookup_csv):
            os.remove(dummy_lookup_csv)
            print("\n> Cleanup complete. Temp file removed.")
