"""
Analysis script for data/az_collab/predict_output.csv

Columns:
  - id:                  variant ID (e.g., 'wt', 'H_S21A', 'L_G45D')
  - binder_seq:          binder sequence as heavy:light chain
  - target_seq:          target protein sequence
  - fitness:             experimental fitness (lower = better; NaN for unlabeled)
  - predicted_affinity:  model prediction (higher = better)

Use --prepare to regenerate data/az_collab/predict_input.csv before analysis.
"""

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------------------
# Configuration
# ---------------------
INPUT_CSV = "data/az_collab/predict_output_untrim.csv"

# Binary threshold: variants with fitness <= this are considered "good binders" (lower = better)
# Default: bottom 25th percentile (fitness <= 25th percentile of labeled data)
BINARY_THRESHOLD = None   # None = auto (top 25%)

# Enrichment factor percents
EF_PERCENTS = [1, 5, 10]
# ---------------------


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "az_collab"


def prepare_input():
    """Regenerate data/az_collab/predict_input.csv from raw files."""
    antigen = ""
    capture = False
    with open(DATA_DIR / "1MLC.fasta") as f:
        for line in f:
            line = line.strip()
            if line == ">1MLC_trim_A":
                capture = True
            elif line.startswith(">"):
                capture = False
            elif capture:
                antigen += line

    seqs = pd.read_csv(DATA_DIR / "d441_seqs.csv", index_col=0)
    seqs.index.name = "id"
    seqs = seqs.reset_index()

    dms = pd.read_csv(DATA_DIR / "d44p1_1mlb_dms_exp_data.csv")
    def normalize_id(s):
        s = re.sub(r'^D44P1_', '', s)
        s = re.sub(r'(\D)0+(\d)', r'\1\2', s)
        return s
    dms["id"] = dms["seq_id"].apply(normalize_id)

    out = seqs.merge(dms[["id", "fitness"]], on="id", how="left")
    out = pd.DataFrame({
        "id": out["id"],
        "binder_seq": out["heavy"] + ":" + out["light"],
        "target_seq": antigen,
        "fitness": out["fitness"],
    })
    out_path = DATA_DIR / "predict_input.csv"
    out.to_csv(out_path, index=False)
    print(f"[Prepare] Saved {len(out)} rows ({out['fitness'].notna().sum()} with fitness) -> {out_path}")


def fmt_p(p):
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.2f}"


def setup_style():
    sns.set_theme(style="whitegrid", rc={
        'axes.edgecolor': '.15',
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
    })
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
        'figure.titlesize': 24,
        'lines.linewidth': 2.5,
    })


def calculate_ef(y_true, y_pred, top_pct, threshold):
    """Enrichment factor: higher pred → higher rank (predicted_affinity is higher-is-better)."""
    n_total = len(y_true)
    n_top = max(1, int(n_total * top_pct / 100))
    actives = (y_true <= threshold)
    total_actives = actives.sum()
    if total_actives == 0:
        return 0.0
    bg_rate = total_actives / n_total
    top_idx = np.argsort(y_pred)[::-1][:n_top]  # descending: high pred first = best predicted binder
    hits = actives.iloc[top_idx].sum()
    return (hits / n_top) / bg_rate


def analyze(csv_path, output_dir):
    print(f"\n[Analysis] {csv_path}")
    df = pd.read_csv(csv_path)
    df['id'] = df['id'].astype(str).str.strip()

    # Split labeled / unlabeled
    labeled = df[df['fitness'].notna()].copy().reset_index(drop=True)
    unlabeled = df[df['fitness'].isna()].copy().reset_index(drop=True)
    print(f"  Labeled: {len(labeled)}  |  Unlabeled: {len(unlabeled)}")

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Correlation metrics (labeled only)
    # ------------------------------------------------------------------
    y_true = labeled['fitness']
    y_pred = labeled['predicted_affinity']
    y_score = y_pred  # higher predicted_affinity = better binder (for AUROC/AUPRC/EF)

    # fitness is lower-is-better; predicted_affinity is higher-is-better → expect negative correlation
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_rho, spearman_p = stats.spearmanr(y_true, y_pred)
    kendall_tau, _ = stats.kendalltau(y_true, y_pred)

    # Binary threshold: lower fitness = better = active → bottom 25%
    global BINARY_THRESHOLD
    if BINARY_THRESHOLD is None:
        threshold = float(np.percentile(y_true, 25))
        print(f"  Auto threshold (25th pctile): {threshold:.2f}")
    else:
        threshold = float(BINARY_THRESHOLD)
        print(f"  Manual threshold: {threshold:.2f}")

    y_bin = (y_true <= threshold).astype(int)
    n_actives = y_bin.sum()
    print(f"  Actives (fitness <= {threshold:.2f}): {n_actives} / {len(labeled)}")

    if len(np.unique(y_bin)) > 1:
        auroc = roc_auc_score(y_bin, y_score)
        auprc = average_precision_score(y_bin, y_score)
    else:
        print("  [Warn] Only one class in binary labels; AUC set to 0.5")
        auroc, auprc = 0.5, 0.0

    efs = {pct: calculate_ef(y_true, y_pred, pct, threshold) for pct in EF_PERCENTS}

    # Print summary
    print("\n" + "=" * 45)
    print(f"  RESULTS: {os.path.basename(csv_path)}")
    print("=" * 45)
    print(f"  Pearson R    : {pearson_r:.4f}  (p={pearson_p:.2e})")
    print(f"  Spearman rho : {spearman_rho:.4f}  (p={spearman_p:.2e})")
    print(f"  Kendall tau  : {kendall_tau:.4f}")
    print(f"\n  Binary threshold : {threshold:.2f}")
    print(f"  AUROC        : {auroc:.4f}")
    print(f"  AUPRC        : {auprc:.4f}")
    for pct, ef in efs.items():
        print(f"  EF @ {pct:2d}%     : {ef:.2f}x")
    print("=" * 45)

    # Save metrics txt
    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"CSV: {csv_path}\n")
        f.write(f"Labeled N: {len(labeled)}\n")
        f.write(f"Unlabeled N: {len(unlabeled)}\n")
        f.write(f"Binary threshold (fitness <=): {threshold:.4f}\n")
        f.write(f"Actives: {n_actives}\n\n")
        f.write(f"Pearson R:    {pearson_r:.4f}  (p={pearson_p:.2e})\n")
        f.write(f"Spearman rho: {spearman_rho:.4f}  (p={spearman_p:.2e})\n")
        f.write(f"Kendall tau:  {kendall_tau:.4f}\n")
        f.write(f"AUROC:        {auroc:.4f}\n")
        f.write(f"AUPRC:        {auprc:.4f}\n")
        for pct, ef in efs.items():
            f.write(f"EF @ {pct}%:     {ef:.4f}x\n")
    print(f"\n  Saved metrics -> {metrics_path}")

    # ------------------------------------------------------------------
    # 2. Scatter plot: predicted_affinity vs fitness (labeled)
    # ------------------------------------------------------------------
    setup_style()
    g = sns.JointGrid(x=y_pred, y=y_true, height=8, ratio=5)
    g.plot_joint(
        sns.regplot,
        color="#D55E00",
        truncate=False,
        scatter_kws={'alpha': 0.5, 's': 60, 'edgecolor': 'white', 'linewidths': 0.6, 'zorder': 2},
        line_kws={'color': "#C44E52", 'alpha': 0.9, 'linewidth': 3, 'zorder': 3},
    )
    g.plot_marginals(sns.kdeplot, color="#D55E00", fill=True, alpha=0.3)
    g.set_axis_labels(xlabel="Predicted Affinity (higher = better)", ylabel="Fitness (lower = better)")
    g.ax_joint.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

    stats_text = (
        f"Spearman ρ = {spearman_rho:.3f} ({fmt_p(spearman_p)})\n"
        f"Pearson R = {pearson_r:.3f} ({fmt_p(pearson_p)})"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='#d3d3d3')
    g.ax_joint.text(0.05, 0.95, stats_text, transform=g.ax_joint.transAxes,
                    verticalalignment='top', bbox=props, fontsize=15, zorder=5)
    g.figure.suptitle("Predicted Affinity vs Experimental Fitness", y=1.01, fontweight='bold')

    scatter_path = os.path.join(output_dir, "scatter_pred_vs_fitness.png")
    plt.savefig(scatter_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved scatter  -> {scatter_path}")

    # ------------------------------------------------------------------
    # 3. Prediction distribution: labeled vs unlabeled
    # ------------------------------------------------------------------
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    sns.kdeplot(labeled['predicted_affinity'], ax=ax, color='#D55E00', fill=True, alpha=0.35,
                label=f'Labeled (n={len(labeled)})', linewidth=2)
    if len(unlabeled) > 0:
        sns.kdeplot(unlabeled['predicted_affinity'], ax=ax, color='#0072B2', fill=True, alpha=0.35,
                    label=f'Unlabeled (n={len(unlabeled)})', linewidth=2)

    ax.set_xlabel("Predicted Affinity")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Predicted Affinity", fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    dist_path = os.path.join(output_dir, "distribution_predictions.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"  Saved dist     -> {dist_path}")

    # ------------------------------------------------------------------
    # 4. Top predictions (unlabeled), sorted by predicted_affinity
    # ------------------------------------------------------------------
    if len(unlabeled) > 0:
        top_unlabeled = unlabeled.sort_values('predicted_affinity', ascending=False).head(50)  # highest = best
        top_path = os.path.join(output_dir, "top50_unlabeled_predictions.csv")
        top_unlabeled[['id', 'predicted_affinity']].to_csv(top_path, index=False)
        print(f"  Saved top-50   -> {top_path}")

    print("\n[Done]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze AZ collab predict_output.csv")
    parser.add_argument("--input", type=str, default=INPUT_CSV,
                        help="Path to predict_output.csv")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same dir as input)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Binary fitness threshold (default: auto 75th pctile)")
    parser.add_argument("--prepare", action="store_true",
                        help="Regenerate data/az_collab/predict_input.csv before analysis")
    args = parser.parse_args()

    if args.prepare:
        prepare_input()

    if args.threshold is not None:
        BINARY_THRESHOLD = args.threshold

    csv_path = args.input
    if not os.path.exists(csv_path):
        print(f"[Error] File not found: {csv_path}")
        sys.exit(1)

    out_dir = args.output_dir or os.path.join(os.path.dirname(csv_path), "analysis")
    analyze(csv_path, out_dir)
