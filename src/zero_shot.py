import sys
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
from transformers import EsmModel, EsmTokenizer

# Fix: Allow importing from 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ----------------------------------------------------------------------
# 1. Zero-Shot Scoring Logic (Bypasses all random heads)
# ----------------------------------------------------------------------
def compute_zero_shot_scores(batch, model, method="cosine"):
    """
    method="cosine": Global Similarity (Higher = Stronger Binder)
    method="energy": Contact Map Energy (Higher = Stronger Binder)
    """
    # Get raw ESM-2 embeddings
    b_out = model(input_ids=batch['binder_ids'], attention_mask=batch['binder_mask'])
    t_out = model(input_ids=batch['target_ids'], attention_mask=batch['target_mask'])
    
    b_raw = b_out.last_hidden_state # (B, L_b, H)
    t_raw = t_out.last_hidden_state # (B, L_t, H)

    if method == "cosine":
        # Mean Pooling of raw embeddings
        b_mask = batch['binder_mask'].unsqueeze(-1).float()
        t_mask = batch['target_mask'].unsqueeze(-1).float()
        
        b_vec = (b_raw * b_mask).sum(1) / b_mask.sum(1).clamp(min=1e-9)
        t_vec = (t_raw * t_mask).sum(1) / t_mask.sum(1).clamp(min=1e-9)
        
        # Cosine similarity: 1.0 (Strong) to -1.0 (Weak)
        score = torch.nn.functional.cosine_similarity(b_vec, t_vec)
        
    elif method == "energy":
        # Interaction Map Energy (Unweighted Dot Product)
        # Normalize for dot-product stability
        b_norm = torch.nn.functional.normalize(b_raw, dim=-1)
        t_norm = torch.nn.functional.normalize(t_raw, dim=-1)
        
        # Raw Attention Map (B, L_b, L_t)
        contact_map = torch.bmm(b_norm, t_norm.transpose(1, 2))
        
        # Mask the padding positions
        mask_2d = torch.bmm(batch['binder_mask'].unsqueeze(-1).float(), 
                           batch['target_mask'].unsqueeze(1).float())
        
        # Average energy of the contact map
        score = (contact_map * mask_2d).sum((1, 2)) / mask_2d.sum((1, 2)).clamp(min=1e-9)

    # REVERSE DIRECTION: 
    # Scores were (Higher = Stronger). 
    # To match log Kd (Higher = Weaker), we negate.
    return -score 

# ----------------------------------------------------------------------
# 2. Main Script
# ----------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[System] Testing ESM-2 Zero-Shot Baseline on {device}")

    # Initialize raw ESM-2 and Tokenizer
    model_name = cfg.model.name # e.g., "facebook/esm2_t33_650M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device).eval()

    # Load Data
    df = pd.read_csv(cfg.input_csv)
    from src.inference import InferenceDataset, InferenceCollator # Use your existing classes
    
    dataloader = DataLoader(
        InferenceDataset(df),
        batch_size=cfg.get("batch_size", 8),
        collate_fn=InferenceCollator(tokenizer, arch="cross_attn", max_length=1024)
    )

    cos_scores, energy_scores = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Method 1: Global Similarity
            c_score = compute_zero_shot_scores(batch, model, method="cosine")
            cos_scores.extend(c_score.cpu().numpy().tolist())
            
            # Method 2: Contact Map Energy
            e_score = compute_zero_shot_scores(batch, model, method="energy")
            energy_scores.extend(e_score.cpu().numpy().tolist())

    df["zero_shot_cosine_logKd"] = cos_scores
    df["zero_shot_energy_logKd"] = energy_scores
    
    # Save Results
    output_path = Path(cfg.input_csv).parent / "zero_shot_results.csv"
    df.to_csv(output_path, index=False)
    print(f"[Success] Zero-Shot benchmarks saved to {output_path}")

if __name__ == "__main__":
    main()
