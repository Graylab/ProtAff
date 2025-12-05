import os
import pandas as pd
import torch
from omegaconf import OmegaConf
from transformers import EsmTokenizer
import hydra

# Import your classes
from dataset import ProteinDataModule

# Mock Hydra path resolution
hydra.utils.to_absolute_path = lambda x: x 

def create_multichain_dummy_data():
    """Creates 3 distinct samples to verify batch processing."""
    
    # 1. Create Lookup
    lookup_data = [
        # --- BINDERS ---
        {"type": "binder", "id": 1, "seq": "MKTV"},               # Sample 0: Single
        {"type": "binder", "id": 2, "seq": "MK:TV"},              # Sample 1: Dual Chain
        {"type": "binder", "id": 3, "seq": "A:B:C:D"},            # Sample 2: 4-Chain Complex
        
        # --- TARGETS ---
        {"type": "target", "id": 1, "seq": "AL"},                 
        {"type": "target", "id": 2, "seq": "AL:KA:EM"},           
        {"type": "target", "id": 3, "seq": "GG"},                 
    ]
    pd.DataFrame(lookup_data).to_csv("dummy_lookup.csv", index=False)

    # 2. Create Base (Pairs)
    base_data = [
        # Sample 0: Single vs Single
        {"binder_id": 1, "target_id": 1, "log_Aff": 0.5, "cluster_id": 1},
        # Sample 1: Multi vs Multi
        {"binder_id": 2, "target_id": 2, "log_Aff": 1.2, "cluster_id": 2},
        # Sample 2: Complex vs Simple
        {"binder_id": 3, "target_id": 3, "log_Aff": 0.9, "cluster_id": 3},
    ]
    pd.DataFrame(base_data).to_csv("dummy_base.csv", index=False)

def check_pooling_logic(ids, tokenizer, stream_name):
    """
    Simulates the pooling mask logic for ALL samples in the batch.
    """
    batch_size = ids.shape[0]
    print(f"\n{'>'*10} Verifying {stream_name} (Batch Size: {batch_size}) {'<'*10}")
    
    # 1. Constants
    cls_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    
    # 2. Create the "Safe Mask" (Logic from model.py)
    # Start: 1 for everything, 0 for PAD
    mask = (ids != pad_id).int() 
    
    # Apply Exclusion Logic: 0 for CLS and 0 for ALL EOS tokens
    pooling_mask = mask.clone()
    pooling_mask[ids == cls_id] = 0
    pooling_mask[ids == eos_id] = 0
    
    # 3. Iterate through EVERY sample
    for i in range(batch_size):
        sample_ids = ids[i]
        sample_mask = pooling_mask[i]
        decoded = tokenizer.convert_ids_to_tokens(sample_ids)
        
        print(f"\n   [{stream_name} - Sample {i}]")
        print(f"   {'TOKEN':<12} | {'ID':<6} | {'MASK':<6} | {'STATUS'}")
        print("   " + "-"*45)
        
        pad_shown = False
        
        for token, tid, m_val in zip(decoded, sample_ids.tolist(), sample_mask.tolist()):
            # Logic Check
            status = "✅ KEEP" if m_val == 1 else "❌ IGNORE"
            
            if tid == pad_id:
                status = "❌ PAD"
                # Logic to only show first PAD line to save space, then skip
                if pad_shown: continue 
                pad_shown = True
                print(f"   {token:<12} | {tid:<6} | {m_val:<6} | {status} (Remaining pads hidden)")
                continue

            if tid == cls_id: status = "❌ CLS"
            if tid == eos_id: status = "❌ EOS"

            print(f"   {token:<12} | {tid:<6} | {m_val:<6} | {status}")

def run_test():
    create_multichain_dummy_data()
    
    # 1. Configuration
    cfg = OmegaConf.create({
        "data": {
            "base_csv": "dummy_base.csv",
            "lookup_csv": "dummy_lookup.csv",
            "weight_col": "cluster_id", 
            "balance_clusters": False,
            "num_workers": 0
        },
        "model": {
            "name": "facebook/esm2_t6_8M_UR50D", 
            "max_length": 128
        },
        "training": {
            "seed": 42,
            "batch_size": 4  # Ensure this fits all 3 samples
        }
    })

    # 2. Setup DataModule
    dm = ProteinDataModule(cfg)
    dm.setup()
    loader = dm.train_dataloader()
    
    # Get the whole batch
    batch = next(iter(loader))
    
    # 3. Verify
    check_pooling_logic(batch['binder_ids'], dm.tokenizer, "BINDER")
    check_pooling_logic(batch['target_ids'], dm.tokenizer, "TARGET")

    # Cleanup
    if os.path.exists("dummy_lookup.csv"): os.remove("dummy_lookup.csv")
    if os.path.exists("dummy_base.csv"): os.remove("dummy_base.csv")

if __name__ == "__main__":
    run_test()