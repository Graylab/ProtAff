"""Save a canonical train/val split to a JSON file.

Usage:
    python src/save_split.py --config-name pair_affinity
    python src/save_split.py --config-name pair_affinity training.seed=42

The output JSON contains:
    {"split_col": "cluster_id", "val_groups": ["grp1", "grp2", ...]}

Set data.split_file in your config to this path to reuse the split.
"""

import json
import sys
import os

import hydra
import pandas as pd
from omegaconf import DictConfig

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.split_utils import group_split


@hydra.main(version_base=None, config_path="../configs")
def main(cfg: DictConfig):
    weight_col = cfg.data.get("weight_col", None)
    split_col = weight_col if weight_col is not None else "target_id"

    base_df = pd.read_csv(cfg.data.base_csv)
    if split_col not in base_df.columns:
        raise KeyError(f"Split column '{split_col}' not found in CSV.")

    seed = cfg.training.get("seed", 42)
    train_ratio = cfg.training.get("train_val_split", 0.9)
    strategy = cfg.training.get("split_strategy", "group")

    if strategy == "random":
        from sklearn.model_selection import train_test_split
        _, val_df = train_test_split(
            base_df, train_size=train_ratio, random_state=seed, shuffle=True
        )
        val_groups = sorted(val_df[split_col].unique().tolist())
    elif strategy == "group":
        _, val_df = group_split(
            base_df, col=split_col, ratio=train_ratio, seed=seed,
            verbose=True, label_col="log_Aff",
        )
        val_groups = sorted(val_df[split_col].unique().tolist())
    else:
        raise ValueError(f"Unknown split_strategy: {strategy}")

    # Convert numpy types to native Python for JSON serialization
    val_groups = [v.item() if hasattr(v, "item") else v for v in val_groups]

    original_cwd = hydra.utils.get_original_cwd()
    split_dir = os.path.join(original_cwd, "data", "splits")
    os.makedirs(split_dir, exist_ok=True)
    out_path = os.path.join(split_dir, f"split_seed{seed}.json")
    split_info = {"split_col": split_col, "val_groups": val_groups}
    with open(out_path, "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nSaved split to: {os.path.abspath(out_path)}")
    print(f"  split_col: {split_col}")
    print(f"  val_groups: {len(val_groups)}")
    print(f"  val_samples: {len(val_df):,}")


if __name__ == "__main__":
    main()
