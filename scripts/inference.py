import sys
import os
import torch
import pandas as pd
import hydra
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader

# Fix: Allow importing from 'src' relative to scripts/
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.model_base import ESMCrossAttentionClassifier
from transformers import EsmTokenizer
from peft import PeftModel

# ----------------------------------------------------------------------
# 1. Inference Dataset
# ----------------------------------------------------------------------
class InferenceDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=1024):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Auto-detect column names
        self.b_col = "binder_sequence" if "binder_sequence" in df.columns else "binder_seq"
        self.t_col = "target_sequence" if "target_sequence" in df.columns else "target_seq"
        
        if self.b_col not in df.columns or self.t_col not in df.columns:
            raise ValueError(f"CSV missing columns. Need '{self.b_col}' and '{self.t_col}'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        b_raw = str(row[self.b_col])
        t_raw = str(row[self.t_col])

        # Logic: Replace ':' with EOS token (Matches training logic for multi-chain)
        b_seq = b_raw.replace(":", self.tokenizer.eos_token)
        t_seq = t_raw.replace(":", self.tokenizer.eos_token)

        # Tokenize with padding/truncation
        b_enc = self.tokenizer(b_seq, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        t_enc = self.tokenizer(t_seq, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")

        return {
            "binder_ids": b_enc["input_ids"].squeeze(0),
            "binder_mask": b_enc["attention_mask"].squeeze(0),
            "target_ids": t_enc["input_ids"].squeeze(0),
            "target_mask": t_enc["attention_mask"].squeeze(0),
        }

# ----------------------------------------------------------------------
# 2. Main Inference Routine
# ----------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    # Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[System] Device: {device}")

    # Paths
    input_path = Path(cfg.input_csv)
    base_res = Path(cfg.base_results_dir)
    save_dir = base_res / cfg.model.name.split("/")[-1] / input_path.stem
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / cfg.output_filename

    # ------------------------------------------------------------------
    # Load Model
    # ------------------------------------------------------------------
    print(f"[Model] Base Architecture: {cfg.model.name}")
    print(f"[Model] Loading Weights: {cfg.model_path}")
    
    if not os.path.exists(cfg.model_path):
        raise FileNotFoundError(f"Model path not found: {cfg.model_path}")

    # A. Init Base (Architecture params from config)
    # Ensure cfg.model.pooling and cfg.model.d_model match training config
    base_model = ESMCrossAttentionClassifier(cfg.model.name, cfg=cfg)
    
    # B. Load Adapter (Standard PEFT loading)
    # Since we are loading the final saved model, keys should match perfectly
    try:
        model = PeftModel.from_pretrained(base_model, cfg.model_path)
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Error loading weights: {e}")
        return

    model.to(device)
    model.eval()
    
    tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------
    df = pd.read_csv(input_path)
    dataset = InferenceDataset(df, tokenizer, max_length=cfg.model.max_length)
    
    # num_workers=0 avoids tokenizer deadlocks and overhead for inference
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    print(f"[Inference] Processing {len(df)} samples...")
    predictions = []

    # ------------------------------------------------------------------
    # Inference Loop
    # ------------------------------------------------------------------
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward Pass
            # We use **batch to unpack dictionary into arguments (binder_ids, etc.)
            logits = model(**batch) 
            
            # Robust Flattening: Ensure output is a 1D list of floats
            # reshape(-1) flattens any (B, 1) or (B) shape to (B)
            flat_scores = logits.reshape(-1).float().cpu().numpy().tolist()
            
            predictions.extend(flat_scores)

    # ------------------------------------------------------------------
    # Save Output
    # ------------------------------------------------------------------
    df["predicted_affinity"] = predictions
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Saved results to: {output_path}")

if __name__ == "__main__":
    main()