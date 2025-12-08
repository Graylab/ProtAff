"""
scripts/analyze_attention.py
Interpretability: Visualizes 'Attentional Pooling' weights to identify binding hotspots.
"""

import sys
import os
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

# --- PATH FIX ---
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.model_base import ESMCrossAttentionClassifier
from transformers import EsmTokenizer
from peft import PeftModel

# ==============================================================================
# VISUALIZATION LOGIC
# ==============================================================================
def plot_attention(seq, weights, title, save_path, color_theme="Blues"):
    """
    Plots a bar chart of attention weights over the sequence.
    """
    # 1. Prepare Data
    # Convert token weights to numpy
    w = weights.detach().cpu().numpy().flatten()
    
    # Filter out special tokens (CLS/EOS) from visualization if they are 0
    # But usually, we just plot valid length.
    # Note: seq string usually lacks CLS/EOS, but tensor has them.
    # We assume 'seq' corresponds to the VALID residues.
    
    # Trim weights to match sequence length (removing padding/special tokens if needed)
    # The pooler mask handles CLS/EOS, so their weights should be 0.
    # We just grab the slice corresponding to the actual seq length.
    # Standard ESM: [CLS] Seq [EOS] [PAD]...
    # Valid weights are at indices 1 to len(seq)+1
    
    valid_weights = w[1 : len(seq) + 1]
    
    # Normalize for visualization (0 to 1 scaling relative to max in this seq)
    if valid_weights.max() > 0:
        norm_weights = valid_weights / valid_weights.max()
    else:
        norm_weights = valid_weights

    # Create DataFrame
    df = pd.DataFrame({
        "Position": range(1, len(seq) + 1),
        "Residue": list(seq),
        "Weight": valid_weights,
        "NormWeight": norm_weights
    })
    
    # Create Labels (e.g. "A1", "R2")
    df["Label"] = df["Residue"] + df["Position"].astype(str)

    # 2. Plot
    plt.figure(figsize=(15, 5))
    sns.set_theme(style="whitegrid")
    
    # Color bars by intensity
    palette = sns.color_palette(color_theme, n_colors=len(df))
    # Sort palette by weight to make high bars darker? 
    # Or just mapping intensity. Let's use a heatmap-bar hybrid.
    
    bars = plt.bar(df["Position"], df["Weight"], color=sns.color_palette(color_theme, as_cmap=True)(df["NormWeight"]))
    
    plt.xlabel("Residue Position")
    plt.ylabel("Attention Weight (Importance)")
    plt.title(title, fontsize=14, weight='bold')
    
    # Highlight Top 5
    top_5 = df.nlargest(5, "Weight")
    for _, row in top_5.iterrows():
        plt.text(row["Position"], row["Weight"], row["Residue"], 
                 ha='center', va='bottom', fontsize=10, weight='bold', color='red')

    # Save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"   -> Plot saved to: {save_path}")
    
    # Print Text Report
    print(f"\n   [Top 5 Hotspots - {title.split(' ')[0]}]")
    for _, row in top_5.iterrows():
        print(f"      Pos {row['Position']} ({row['Residue']}): {row['Weight']:.4f}")

# ==============================================================================
# MANUAL FORWARD PASS (To extract intermediate weights)
# ==============================================================================
def get_attention_weights(model, batch, device):
    """
    Runs the model layer-by-layer to get the pooling weights.
    """
    bm = model.base_model.model # Access inner ESMCrossAttentionClassifier
    
    # Move batch to device
    b_ids = batch['binder_ids'].to(device)
    b_mask = batch['binder_mask'].to(device)
    t_ids = batch['target_ids'].to(device)
    t_mask = batch['target_mask'].to(device)

    with torch.no_grad():
        # 1. ESM
        b_raw = bm.esm(b_ids, attention_mask=b_mask).last_hidden_state
        t_raw = bm.esm(t_ids, attention_mask=t_mask).last_hidden_state
        
        # 2. Project
        b_vec = bm.norm_input(bm.projector(b_raw))
        t_vec = bm.norm_input(bm.projector(t_raw))
        
        # 3. Encoder
        b_key_mask = ~b_mask.bool()
        t_key_mask = ~t_mask.bool()
        b_enc = bm.task_encoder(b_vec, src_key_padding_mask=b_key_mask)
        t_enc = bm.task_encoder(t_vec, src_key_padding_mask=t_key_mask)
        
        # 4. Decoder
        dec_b = bm.decoder_b2t(b_enc, memory=t_enc, tgt_key_padding_mask=b_key_mask, memory_key_padding_mask=t_key_mask)
        dec_t = bm.decoder_t2b(t_enc, memory=b_enc, tgt_key_padding_mask=t_key_mask, memory_key_padding_mask=b_key_mask)
        
        # 5. POOLING (Manual Calculation to extract weights)
        # We need to replicate AttentionalPooling logic here to get the 'weights' tensor
        
        def calculate_weights(pooler, x, mask_ids, attn_mask):
            # Create mask (remove CLS/EOS/PAD)
            p_mask = bm.get_pooling_mask(mask_ids, attn_mask) # (B, L, 1)
            
            # Linear Proj -> Tanh -> Linear
            scores = pooler.attn(x) # (B, L, 1)
            
            # Masking
            scores = scores.masked_fill(p_mask == 0, -1e9)
            
            # Softmax -> These are the weights we want!
            weights = torch.softmax(scores, dim=1)
            return weights

        # Check if attention pooling is enabled
        if bm.pooling_type == "attention":
            w_b = calculate_weights(bm.pooler_b, dec_b, b_ids, b_mask)
            w_t = calculate_weights(bm.pooler_t, dec_t, t_ids, t_mask)
            
            # Predict Affinity just for sanity check
            # (Weighted Sum)
            pool_b = (dec_b * w_b).sum(dim=1)
            pool_t = (dec_t * w_t).sum(dim=1)
            fused = torch.cat([bm.norm_pooled_b(pool_b), bm.norm_pooled_t(pool_t)], dim=-1)
            pred = bm.head_score(fused).item()
            
            return w_b, w_t, pred
        else:
            print("❌ Error: Model is using Mean Pooling. Cannot visualize attention weights.")
            return None, None, 0.0

@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[System] Device: {device}")
    
    # Output Setup
    out_dir = Path("analysis_output/attention_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load Model
    # ------------------------------------------------------------------
    print(f"[Model] Loading from: {cfg.model_path}")
    base_model = ESMCrossAttentionClassifier(cfg.model.name, cfg=cfg)
    try:
        model = PeftModel.from_pretrained(base_model, cfg.model_path)
    except Exception as e:
        print(f"❌ Error loading: {e}")
        return
    
    model.to(device)
    model.eval()
    tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)

    # ------------------------------------------------------------------
    # 2. Select Sample (From CSV or Manual)
    # ------------------------------------------------------------------
    # Option A: Manual Strings (Edit here for quick checks)
    binder_seq = "QVQLQESGPGLVKPSETLSLTCTVSGGSVSSGDYYWTWIRQSPGKGLEWIGHIYYSGNTNYNPSLKSRLTISIDTSKTQFSLKLSSVTAADTAVYYCVRDRTLYGMDVWGQGTTVTVSS"
    target_seq = "TVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    
    # Option B: From CSV (Uncomment to use first row of input csv)
    if cfg.input_csv:
        df = pd.read_csv(cfg.input_csv)
        b_col = "binder_sequence" if "binder_sequence" in df.columns else "binder_seq"
        t_col = "target_sequence" if "target_sequence" in df.columns else "target_seq"
        binder_seq = df.iloc[0][b_col]
        target_seq = df.iloc[0][t_col]
        print(f"[Data] Using first sample from {cfg.input_csv}")

    # Clean
    binder_seq = str(binder_seq).replace(":", tokenizer.eos_token)
    target_seq = str(target_seq).replace(":", tokenizer.eos_token)

    print(f"Binder: {binder_seq[:20]}... ({len(binder_seq)} AA)")
    print(f"Target: {target_seq[:20]}... ({len(target_seq)} AA)")

    # ------------------------------------------------------------------
    # 3. Process
    # ------------------------------------------------------------------
    b_enc = tokenizer(binder_seq, return_tensors="pt")
    t_enc = tokenizer(target_seq, return_tensors="pt")
    
    batch = {
        "binder_ids": b_enc["input_ids"],
        "binder_mask": b_enc["attention_mask"],
        "target_ids": t_enc["input_ids"],
        "target_mask": t_enc["attention_mask"]
    }

    # Get Weights
    weights_b, weights_t, pred_affinity = get_attention_weights(model, batch, device)

    if weights_b is not None:
        print(f"\n[Result] Predicted Affinity: {pred_affinity:.4f}")
        
        # ------------------------------------------------------------------
        # 4. Plot
        # ------------------------------------------------------------------
        # Binder Plot
        plot_attention(
            binder_seq.replace(tokenizer.eos_token, ""), # Remove EOS for clean plotting
            weights_b[0], 
            f"Binder Attention (Hotspots) - Pred: {pred_affinity:.2f}",
            out_dir / "binder_attention.png",
            color_theme="Reds"
        )
        
        # Target Plot
        plot_attention(
            target_seq.replace(tokenizer.eos_token, ""),
            weights_t[0], 
            f"Target Attention (Hotspots)",
            out_dir / "target_attention.png",
            color_theme="Blues"
        )
        
        print("\n✅ Analysis Complete. Check 'analysis_output/attention_plots/'")

if __name__ == "__main__":
    main()
