import sys
import os
import pytest
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from transformers import EsmTokenizer

# --- FIX: Add project root to path BEFORE importing src ---
# This ensures Python finds 'src' regardless of where you run the test from
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import your classes
from src.datasets.dataset_affinity import AffinityDataModule, ConcatCollator, AffinityDataset

# ------------------------------------------------------------------
# 1. Setup Dummy Data Fixture
# ------------------------------------------------------------------
@pytest.fixture
def mock_data(tmp_path):
    """Creates temporary dummy CSVs for testing."""
    
    # A. Lookup CSV (Sequence Storage)
    lookup_data = {
        "type": ["binder", "binder", "target", "target"],
        "id": [101, 102, 201, 202],
        "seq": [
            "MKTLLILAVSLIA",           # Short Binder
            "QVQLVQSGAEVKKPGASVKV",    # Antibody-like Binder
            "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDAPKLRMVSAV", # Long Target
            "ACDEFGHIKLMNPQRSTVWY"     # Short Target
        ]
    }
    lookup_df = pd.DataFrame(lookup_data)
    lookup_path = tmp_path / "lookup.csv"
    lookup_df.to_csv(lookup_path, index=False)

    # B. Base CSV (Affinity Data)
    # Case 1: log_Aff < 5.0 -> Binder (1.0)
    # Case 2: log_Aff >= 5.0 -> Non-Binder (0.0)
    # Case 3: log_Aff is NaN -> Masked (-100.0)
    base_data = {
        "binder_id": [101, 102, 101, 102],
        "target_id": [201, 201, 202, 202],
        "log_Aff": [4.5, 9.0, np.nan, 2.0], 
        "cluster_id": [1, 1, 2, 2]
    }
    base_df = pd.DataFrame(base_data)
    base_path = tmp_path / "affinity_train.csv"
    base_df.to_csv(base_path, index=False)

    return base_path, lookup_path

# ------------------------------------------------------------------
# 2. Test Dataset Logic
# ------------------------------------------------------------------
def test_dataset_initialization(mock_data):
    base_path, lookup_path = mock_data
    
    # Initialize Dataset
    ds = AffinityDataset(
        pd.read_csv(base_path), 
        str(lookup_path), 
        weight_col="cluster_id", 
        balance_clusters=True
    )

    assert len(ds) == 4
    
    # Case 1: log_Aff = 4.5 (< 5.0) -> Should be Binder (1.0)
    item0 = ds[0]
    assert item0["log_Aff"] == 4.5
    assert item0["is_binder"] == 1.0 
    
    # Case 2: log_Aff = 9.0 (>= 5.0) -> Should be Non-Binder (0.0)
    item1 = ds[1]
    assert item1["log_Aff"] == 9.0
    assert item1["is_binder"] == 0.0

    # Case 3: log_Aff = NaN -> is_binder should default to 0.0 (per np.where logic)
    item2 = ds[2]
    assert np.isnan(item2["log_Aff"])
    assert item2["is_binder"] == 0.0

# ------------------------------------------------------------------
# 3. Test Collator Structure & Masking
# ------------------------------------------------------------------
def test_collator_structure(mock_data):
    base_path, lookup_path = mock_data
    
    # Use ESM tokenizer (Standard)
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    
    # Config
    cfg = OmegaConf.create({
        "model": {"name": "facebook/esm2_t6_8M_UR50D", "max_length": 128},
        "data": {
            "base_csv": str(base_path),
            "lookup_csv": str(lookup_path),
            "num_workers": 0,
            "weight_col": "cluster_id",
            "balance_clusters": False
        },
        "training": {"batch_size": 2, "seed": 42}
    })

    dm = AffinityDataModule(cfg)
    dm.setup()
    
    # Get a batch
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    
    # A. Check Keys
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "cls_labels" in batch
    assert "reg_labels" in batch
    
    # B. Check Shapes
    B, L = batch["input_ids"].shape
    assert B == 2
    assert L <= 128
    
    # C. Check Special Token Pattern: [CLS] ... [EOS] ... [EOS]
    cls_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    
    for i in range(B):
        seq = batch["input_ids"][i]
        mask = batch["attention_mask"][i]
        
        # 1. Start with CLS
        assert seq[0] == cls_id
        
        # 2. Count EOS in the valid (non-padded) region
        valid_len = mask.sum()
        valid_seq = seq[:valid_len]
        assert (valid_seq == eos_id).sum() == 2, "Must contain exactly two EOS tokens in valid region"
        
        # 3. Verify Padding
        if L > valid_len:
            assert (seq[valid_len:] == pad_id).all()

    # D. Check Regression Masking (NaN -> -100.0)
    reg_labels = batch["reg_labels"]
    
    # Let's manually invoke collator with a NaN feature to be sure
    collator = ConcatCollator(tokenizer, max_length=128)
    nan_feature = [{"binder_seq": "A", "target_seq": "A", "log_Aff": np.nan, "is_binder": 0.0}]
    nan_batch = collator(nan_feature)
    
    assert nan_batch["reg_labels"][0] == -100.0

# ------------------------------------------------------------------
# 4. Test Smart Truncation
# ------------------------------------------------------------------
def test_truncation_logic():
    """
    Test that [CLS] Binder [EOS] Target [EOS] respects max_length 
    and truncates Target first.
    """
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    
    # Tiny max_length to force truncation
    # Budget = 10 - 3 specials = 7 tokens for sequences
    collator = ConcatCollator(tokenizer, max_length=10) 
    
    # Case: Binder (5) + Target (5) = 10 tokens. 
    # Total needed: 1 + 5 + 1 + 5 + 1 = 13 tokens.
    # Excess: 3 tokens.
    # Logic: Target should be cut by 3.
    # Expected: Binder (5) + Target (2)
    features = [{
        "binder_seq": "AAAAA", 
        "target_seq": "TTTTT",
        "is_binder": 1.0,
        "log_Aff": 2.0
    }]
    
    batch = collator(features)
    input_ids = batch["input_ids"][0]
    
    # 1. Check Length
    assert len(input_ids) == 10
    
    # 2. Check Structure
    # Indices: 0=CLS, 1-5=A, 6=EOS, 7-8=T, 9=EOS
    cls_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    
    assert input_ids[0] == cls_id
    assert input_ids[6] == eos_id
    assert input_ids[9] == eos_id
    
    # 3. Verify Content
    valid_tokens = input_ids[1:6] # Binder
    # Assuming 'A' maps to a single token (it does in ESM)
    assert len(valid_tokens) == 5 
    
    valid_target = input_ids[7:9] # Target
    assert len(valid_target) == 2 

    print("\n[Test] Smart Truncation verified: Target truncated to preserve Binder.")

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))