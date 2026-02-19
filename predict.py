#!/usr/bin/env python
"""
Standalone prediction script for ProtAff models.

Usage:
    python predict.py --model_dir /path/to/saved_model --input_csv data.csv --output_csv results.csv

    # Override architecture (if model_config.yaml is not in saved_model/)
    python predict.py --model_dir /path/to/saved_model --input_csv data.csv --output_csv results.csv \
        --arch binding --d_model 256 --n_heads 8 --n_cross_layers 2

Input CSV must have binder and target sequence columns. Accepted column names:
    binder: binder_sequence, binder_seq, binder, seq_1, heavy_chain, cdr3
    target: target_sequence, target_seq, target, seq_2, antigen
"""
import argparse
import sys
import os
import yaml
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Allow imports from the ProtAff project
PROTAFF_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROTAFF_ROOT))

from src.models import build_model
from src.datasets.collators import CROSS_ATTN_ARCHS, select_collator
from transformers import EsmTokenizer
from peft import PeftModel
from omegaconf import OmegaConf


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
BINDER_CANDIDATES = ["binder_sequence", "binder_seq", "binder", "seq_1", "heavy_chain", "cdr3"]
TARGET_CANDIDATES = ["target_sequence", "target_seq", "target", "seq_2", "antigen"]


class InferenceDataset(Dataset):
    def __init__(self, df, b_col, t_col):
        self.records = df[[b_col, t_col]].rename(
            columns={b_col: "binder_seq", t_col: "target_seq"}
        ).to_dict("records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def find_col(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
def load_model(model_dir, model_cfg, device):
    base_model = build_model(model_cfg)
    model = PeftModel.from_pretrained(base_model, model_dir)
    return model.to(device).eval()


def build_cfg_from_args(args):
    """Build an OmegaConf config compatible with build_model()."""
    model_dir = Path(args.model_dir)
    config_path = model_dir / "model_config.yaml"

    if config_path.exists():
        print(f"[Config] Loading from {config_path}")
        model_dict = yaml.safe_load(config_path.read_text())
    else:
        print(f"[Config] No model_config.yaml found, using CLI args / defaults")
        model_dict = {
            "name": args.esm_model,
            "arch": args.arch,
            "max_length": args.max_length,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_cross_layers": args.n_cross_layers,
            "dropout": args.dropout,
        }
        if args.arch == "binding":
            model_dict["binding_head"] = args.binding_head

    return OmegaConf.create({"model": model_dict})


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------
def run_inference(model, dataloader, arch, device, binding_head="affinity"):
    is_cross_attn = arch in CROSS_ATTN_ARCHS
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            batch = {k: v.to(device) for k, v in batch.items()}

            if is_cross_attn:
                kwargs = dict(
                    binder_ids=batch["binder_ids"], binder_mask=batch["binder_mask"],
                    target_ids=batch["target_ids"], target_mask=batch["target_mask"],
                )
                if arch == "binding":
                    kwargs["task"] = binding_head
                outputs = model(**kwargs)
            else:
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

            predictions.extend(outputs.reshape(-1).float().cpu().numpy().tolist())

    return predictions


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ProtAff prediction")
    parser.add_argument("--model_dir", required=True, help="Path to saved_model/ directory")
    parser.add_argument("--input_csv", required=True, help="Input CSV with binder/target sequences")
    parser.add_argument("--output_csv", required=True, help="Output CSV path for predictions")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    # Model config overrides (used only if model_config.yaml is absent)
    parser.add_argument("--arch", default="binding",
                        choices=["concat", "cross_attn", "bi_cross_attn", "binding"])
    parser.add_argument("--esm_model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_cross_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--binding_head", default="affinity",
                        choices=["affinity", "classify"])
    parser.add_argument("--target_seq", default=None,
                        help="Target sequence applied to all rows. "
                             "If provided, the input CSV does not need a target column.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = build_cfg_from_args(args)
    arch = cfg.model.arch
    print(f"[System] Device: {device} | Architecture: {arch}")

    # Load model
    model = load_model(args.model_dir, cfg, device)
    tokenizer = EsmTokenizer.from_pretrained(cfg.model.name)

    # Load data
    df = pd.read_csv(args.input_csv)
    b_col = find_col(df.columns, BINDER_CANDIDATES)
    if not b_col:
        print(f"[ERROR] Could not find binder column in CSV.")
        print(f"  Found columns: {list(df.columns)}")
        print(f"  Expected binder: {BINDER_CANDIDATES}")
        sys.exit(1)

    # Resolve target column: --target_seq flag takes priority over CSV column
    if args.target_seq:
        df["target_sequence"] = args.target_seq
        t_col = "target_sequence"
    else:
        t_col = find_col(df.columns, TARGET_CANDIDATES)
        if not t_col:
            print(f"[ERROR] Could not find target column in CSV and --target_seq not provided.")
            print(f"  Found columns: {list(df.columns)}")
            print(f"  Expected target: {TARGET_CANDIDATES}")
            sys.exit(1)
    print(f"[Data] Using columns: binder={b_col}, target={t_col}, rows={len(df)}")

    max_length = cfg.model.get("max_length", 2048 if arch == "concat" else 1024)
    dataloader = DataLoader(
        InferenceDataset(df, b_col, t_col),
        batch_size=args.batch_size,
        collate_fn=select_collator(arch, tokenizer, max_length, mode="inference"),
        num_workers=args.num_workers,
    )

    # Run inference
    binding_head = cfg.model.get("binding_head", "affinity")
    predictions = run_inference(model, dataloader, arch, device, binding_head=binding_head)

    # Save results
    df["predicted_affinity"] = predictions
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] {len(predictions)} predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
