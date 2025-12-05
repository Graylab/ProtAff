import argparse
import pandas as pd
import torch
from omegaconf import OmegaConf
import hydra
import difflib
import os

# Import your module
try:
    from dataset_dms import DMSDataModule
except ImportError:
    print("❌ Error: Could not import 'DMSDataModule'.")
    exit(1)

# Mock Hydra path resolution
hydra.utils.to_absolute_path = lambda x: x 

def compare_sequences(seq1, seq2):
    """
    Smart comparison that handles Indels (Insertions/Deletions).
    """
    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    diffs = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Substitution (e.g. A -> G)
            segment_wt = seq1[i1:i2]
            segment_mut = seq2[j1:j2]
            diffs.append(f"Sub({i1}): {segment_wt}->{segment_mut}")
        elif tag == 'delete':
            # Deletion (WT has it, Mut lost it)
            segment = seq1[i1:i2]
            diffs.append(f"Del({i1}): {segment}")
        elif tag == 'insert':
            # Insertion (WT didn't have it, Mut added it)
            segment = seq2[j1:j2]
            diffs.append(f"Ins({i1}): {segment}")
            
    if not diffs:
        return "Identical sequences"
        
    # Limit output to first 5 changes to keep logs clean
    if len(diffs) > 5:
        return ", ".join(diffs[:5]) + " ... (more)"
    return ", ".join(diffs)

def verify_data(mutant_csv, wt_csv, limit=None):
    temp_mutant_file = "temp_debug_mutants.csv"
    use_mutant_path = mutant_csv

    # 1. Handle Subsetting (Mutants Only)
    if limit:
        print(f"\n✂️  Loading top {limit} rows from Mutant CSV...")
        try:
            # Read only N rows
            df_m = pd.read_csv(mutant_csv, nrows=limit)
            
            # Save temp file
            df_m.to_csv(temp_mutant_file, index=False)
            use_mutant_path = temp_mutant_file
            print(f"   -> Temporary subset created: {temp_mutant_file}")
        except Exception as e:
            print(f"❌ Error creating subset: {e}")
            return

    # 2. Config
    # We use the ORIGINAL wt_csv, but the (possibly temp) mutant_csv
    cfg = OmegaConf.create({
        "data": {
            "mutant_csv": use_mutant_path,
            "wt_csv": wt_csv,
            "num_workers": 0 
        },
        "model": {
            "name": "facebook/esm2_t6_8M_UR50D", 
            "max_length": 512 
        },
        "training": {
            "seed": 42,
            "batch_size": 4 
        }
    })

    # 3. Initialize & Setup
    print("\n[Initializing DataModule]...")
    dm = DMSDataModule(cfg)
    try:
        dm.setup()
    except Exception as e:
        print(f"❌ Error during setup(): {e}")
        # Cleanup before exiting
        if limit and os.path.exists(temp_mutant_file): os.remove(temp_mutant_file)
        return

    # 4. Fetch Batch
    print("[Fetching Batch]...")
    try:
        loader = dm.train_dataloader()
        batch = next(iter(loader))
    except StopIteration:
        print("❌ Error: DataLoader is empty. Check if filenames match between CSVs.")
        if limit and os.path.exists(temp_mutant_file): os.remove(temp_mutant_file)
        return

    # 5. Inspect
    tokenizer = dm.tokenizer
    wt_ids = batch['wt_ids']
    mut_ids = batch['mut_ids']
    labels = batch['labels']

    print("\n" + "="*60)
    print(f"BATCH INSPECTION (Batch Size: {len(labels)})")
    print("="*60)
    
    for i in range(min(len(wt_ids), 4)):
        print(f"\nSample {i}:")
        
        # Decode
        wt_seq = tokenizer.decode(wt_ids[i], skip_special_tokens=True).replace(" ", "")
        mut_seq = tokenizer.decode(mut_ids[i], skip_special_tokens=True).replace(" ", "")
        
        label_val = labels[i].item()
        
        print(f"   Label (Delta):     {label_val:.4f}")
        print(f"   Mutation Analysis: {compare_sequences(wt_seq, mut_seq)}")
        print(f"   WT Sequence:       {wt_seq[:20]}...")
        
    print("\n✅ Verification Complete.")
    
    # 6. Cleanup
    if limit and os.path.exists(temp_mutant_file):
        os.remove(temp_mutant_file)
        print("   -> Temporary file removed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutant", type=str, required=True)
    parser.add_argument("--wt", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Rows to load from Mutant CSV")
    
    args = parser.parse_args()
    
    verify_data(args.mutant, args.wt, args.limit)