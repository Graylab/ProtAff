import sys
import os
import torch
import pandas as pd
import hydra
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Fix: Allow importing from 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models import build_model
from transformers import EsmTokenizer
from peft import PeftModel

# ----------------------------------------------------------------------
# 1. Inference Dataset & Collator
# ----------------------------------------------------------------------
class InferenceDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
        # Helper to find columns
        def find_col(candidates):
            for c in candidates:
                if c in self.df.columns: return c
            return None

        # Robust Column Detection
        self.b_col = find_col(["binder_sequence", "binder_seq", "binder", "seq_1", "heavy_chain", "cdr3"])
        self.t_col = find_col(["target_sequence", "target_seq", "target", "seq_2", "antigen"])
        
        if not self.b_col or not self.t_col:
            raise ValueError(f"CSV missing columns. Need variants of 'binder_sequence' and 'target_sequence'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "binder_seq": str(row[self.b_col]),
            "target_seq": str(row[self.t_col])
        }

class InferenceCollator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        eos = self.tokenizer.eos_token
        
        # 1. Tokenize
        binder_seqs = [x["binder_seq"].replace(":", eos) for x in batch]
        target_seqs = [x["target_seq"].replace(":", eos) for x in batch]
        
        b_encoded = self.tokenizer(binder_seqs, add_special_tokens=False)
        t_encoded = self.tokenizer(target_seqs, add_special_tokens=False)
        
        input_ids_list = []
        attention_mask_list = []
        
        cls_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        
        # 2. Concat & Truncate
        for b_ids, t_ids in zip(b_encoded["input_ids"], t_encoded["input_ids"]):
            budget = self.max_length - 3
            curr_len = len(b_ids) + len(t_ids)
            
            if curr_len > budget:
                remaining = budget - len(b_ids)
                if remaining > 0:
                    t_ids = t_ids[:remaining]
                else:
                    b_ids = b_ids[:budget]
                    t_ids = []
            
            full_ids = [cls_id] + b_ids + [eos_id] + t_ids + [eos_id]
            
            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            attention_mask_list.append(torch.ones(len(full_ids), dtype=torch.long))

        # 3. Dynamic Pad
        batch_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        batch_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        
        return {
            "input_ids": batch_ids,
            "attention_mask": batch_mask
        }

# ----------------------------------------------------------------------
# 2. Main Inference Routine
# ----------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    # Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[System] Device: {device}")

    input_path = Path(cfg.input_csv)
    
    # --- AUTO-TAG LOGIC ---
    if cfg.get("tag"):
        exp_tag = cfg.tag
    else:
        path_obj = Path(cfg.model_path)
        if path_obj.is_file():
            path_obj = path_obj.parent

        generic_names = ["saved_model", "checkpoints", "best_model", "last_model"]
        
        while path_obj.name in generic_names or path_obj.name.startswith("checkpoint-"):
            path_obj = path_obj.parent
            
        parent_name = path_obj.parent.name
        if len(parent_name) == 10 and parent_name.count("-") == 2 and parent_name[0] == '2':
             exp_tag = f"{parent_name}_{path_obj.name}"
        else:
             exp_tag = path_obj.name
            
    print(f"[System] Experiment Tag Inferred: {exp_tag}")

    # --- OUTPUT PATH LOGIC (DATASET -> VERSION) ---
    if cfg.get("base_results_dir"):
        result_root = Path(cfg.base_results_dir) / exp_tag
    else:
        # e.g. outputs/inference/2025-12-29_02-20-21
        result_root = Path("outputs/inference") / exp_tag
        
    base_folder_name = input_path.stem # e.g. "test_binder"
    dataset_dir = result_root / base_folder_name
    
    # Find next available 'vX' folder, starting from v0
    # Path: .../2025-12-29.../test_binder/v0/predictions.csv
    counter = 0
    while True:
        save_dir = dataset_dir / f"v{counter}"
        
        if not save_dir.exists():
            break
        counter += 1
            
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / cfg.get("output_filename", "predictions.csv")
    
    print(f"[System] Output Directory: {save_dir}")
    # ---------------------------------------------

    print(f"[Model] Base Architecture: {cfg.model.name}")
    print(f"[Model] Loading Weights: {cfg.model_path}")
    
    if not os.path.exists(cfg.model_path):
        raise FileNotFoundError(f"Model path not found: {cfg.model_path}")

    # A. Init Base
    base_model = build_model(cfg)
    
    # B. Load Adapter
    try:
        model = PeftModel.from_pretrained(base_model, cfg.model_path)
        print("[INFO] Model loaded successfully (Adapters Injected).")
    except Exception as e:
        print(f"[ERROR] Error loading weights: {e}")
        return

    model.to(device)
    model.eval()
    
    tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)

    # Data Loading
    df = pd.read_csv(input_path)
    dataset = InferenceDataset(df)
    collator = InferenceCollator(tokenizer, max_length=cfg.model.get("max_length", 1024))
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.get("batch_size", 16), 
        shuffle=False, 
        num_workers=cfg.get("num_workers", 0),
        collate_fn=collator 
    )

    print(f"[Inference] Processing {len(df)} samples...")
    
    predictions_reg = []

    # Inference Loop
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward Pass (Scalar Output)
            outputs = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            
            flat_scores = outputs.reshape(-1).float().cpu().numpy().tolist()
            predictions_reg.extend(flat_scores)

    # Save Output
    df["predicted_affinity"] = predictions_reg
    
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Saved results to: {output_path}")

if __name__ == "__main__":
    main()