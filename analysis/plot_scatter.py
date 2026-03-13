import sys
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    roc_auc_score, 
    average_precision_score
)

# ---------------------
# Configuration
# ---------------------

# ---------------------

def calculate_enrichment_factor(y_true, y_pred, top_percent, binary_threshold):
    """
    Calculates EF. y_true is -log_Aff (higher = better binder).
    Predictions are sorted descending (highest predicted affinity first).
    """
    n_total = len(y_true)
    n_top = int(n_total * (top_percent / 100.0))
    if n_top == 0: n_top = 1

    # 1. Identify Actives (True Binders)
    actives_mask = y_true >= binary_threshold
    total_actives = actives_mask.sum()
    background_rate = total_actives / n_total

    if total_actives == 0: return 0.0

    # 2. Sort Predictions Descending (Highest predicted affinity = best rank)
    sorted_indices = np.argsort(y_pred)[::-1]

    # 3. Select Top K
    top_indices = sorted_indices[:n_top]

    # 4. Count Hits
    hits_in_top = actives_mask[top_indices].sum()

    # 5. Calculate EF
    selection_rate = hits_in_top / n_top
    return selection_rate / background_rate

def analyze_and_plot(csv_path, gt_col="log_Aff", threshold=None):
    print(f"\n[Analysis] Processing: {csv_path}")
    
    # 1. Load Data
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if gt_col not in df.columns or "predicted_affinity" not in df.columns:
        print(f"[Error] CSV must contain '{gt_col}' and 'predicted_affinity'. Found: {df.columns.tolist()}")
        return

    # Drop rows where ground truth or prediction is NaN
    mask = df[gt_col].notna() & df["predicted_affinity"].notna()
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        print(f"[Info] Dropped {n_dropped} rows with NaN in '{gt_col}' or 'predicted_affinity'")
    df = df[mask]

    y_true = -df[gt_col].values
    y_pred = -df["predicted_affinity"].values
    
    output_dir = os.path.dirname(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    # -----------------------------------------------------------
    # 2. Standard Metrics (Calculated on RAW values)
    # -----------------------------------------------------------
    pearson_r, _ = stats.pearsonr(y_true, y_pred)
    spearman_rho, _ = stats.spearmanr(y_true, y_pred)
    kendall_tau, _ = stats.kendalltau(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # -----------------------------------------------------------
    # 3. Design Metrics (Enrichment & Classification)
    # -----------------------------------------------------------
    
    # Logic: Determine Threshold
    if threshold is not None:
        used_threshold = -threshold  # negate to match y_true space
        print(f"[Config] Using MANUAL threshold: {threshold} (negated: {used_threshold})")
    else:
        used_threshold = np.percentile(y_true, 90)
        print(f"[Config] Using calculated 90th percentile: {used_threshold:.2f}")

    # Logic: Define Actives based on threshold (-gt_col >= threshold = good binder)
    print(f"[Config] 'Good' defined as -{gt_col} >= {used_threshold:.2f}")
    y_bin = (y_true >= used_threshold).astype(int)
    y_score_for_auc = y_pred

    # AUC Scores
    if len(np.unique(y_bin)) > 1:
        auroc = roc_auc_score(y_bin, y_score_for_auc)
        auprc = average_precision_score(y_bin, y_score_for_auc)
    else:
        print("[Warn] Only one class present in binary labels. AUC set to 0.5")
        auroc, auprc = 0.5, 0.0

    # Enrichment Factors
    ef_1 = calculate_enrichment_factor(y_true, y_pred, 1.0, used_threshold)
    ef_5 = calculate_enrichment_factor(y_true, y_pred, 5.0, used_threshold)
    ef_10 = calculate_enrichment_factor(y_true, y_pred, 10.0, used_threshold)

    # Console Output
    print("-" * 40)
    print(f"RESULTS: {os.path.basename(csv_path)}")
    print("-" * 40)
    print(f"--- Correlation ---")
    print(f"Pearson R    : {pearson_r:.4f}")
    print(f"Spearman Rho : {spearman_rho:.4f}")
    print(f"Kendall Tau  : {kendall_tau:.4f}")
    print(f"\n--- Error ---")
    print(f"RMSE         : {rmse:.4f}")
    print(f"MAE          : {mae:.4f}")
    print(f"\n--- Screening Power (Threshold: {used_threshold:.2f}) ---")
    print(f"AUROC        : {auroc:.4f}")
    print(f"AUPRC        : {auprc:.4f}")
    print(f"EF @ 1%      : {ef_1:.2f}x")
    print(f"EF @ 5%      : {ef_5:.2f}x")
    print(f"EF @ 10%     : {ef_10:.2f}x")
    print("-" * 40)

    # Save Metrics
    txt_path = os.path.join(output_dir, f"{base_name}_metrics.txt")
    with open(txt_path, "w") as f:
        f.write(f"Threshold used: {used_threshold}\n")
        f.write(f"Pearson R: {pearson_r:.4f}\n")
        f.write(f"Spearman Rho: {spearman_rho:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"AUROC: {auroc:.4f}\n")
        f.write(f"EF 1%: {ef_1:.4f}\n")
        f.write(f"EF 10%: {ef_10:.4f}\n")

    # -----------------------------------------------------------
    # 4. Plot (NEGATED values)
    # -----------------------------------------------------------
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
        'axes.labelsize': 24,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
        'figure.titlesize': 26,
        'lines.linewidth': 2.5,
    })

    # x = predicted_affinity (higher=better), y = -log_Aff (higher=better)
    x_plot = y_pred
    y_plot = y_true

    g = sns.JointGrid(x=x_plot, y=y_plot, height=8, ratio=5)

    g.plot_joint(sns.regplot,
                 color="#D55E00",
                 truncate=False,
                 scatter_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'white', 'linewidths': 0.8, 'zorder': 2},
                 line_kws={'color': "#C44E52", 'alpha': 0.9, 'linewidth': 3, 'label': 'Linear Fit', 'zorder': 3})

    g.plot_marginals(sns.kdeplot, color="#D55E00", fill=True, alpha=0.3)

    g.set_axis_labels(xlabel="-Predicted Affinity (Higher is Better)", ylabel=f"-{gt_col} (Higher is Better)")
    g.ax_joint.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color='gray')

    stats_text = (
        f"Spearman: {spearman_rho:.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"EF@10%: {ef_10:.1f}x\n"
        f"AUC: {auroc:.2f}"
    )

    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='#d3d3d3')
    g.ax_joint.text(0.05, 0.95, stats_text,
                    transform=g.ax_joint.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=props, fontsize=16, zorder=5)

    g.ax_joint.legend(loc='upper right', frameon=True, framealpha=0.9)

    plot_path = os.path.join(output_dir, f"{base_name}_scatter.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"[Success] Plot saved to: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze inference results.")
    parser.add_argument("path", type=str, help="Path to csv file OR directory containing predictions.csv")
    parser.add_argument("--gt-col", type=str, default="log_Aff", help="Ground truth column name (default: log_Aff)")
    parser.add_argument("--threshold", type=float, default=None, help="Binary threshold in original scale (default: 90th percentile of negated values)")

    args = parser.parse_args()

    target_path = args.path

    # Auto-infer logic
    if os.path.isdir(target_path):
        # If it's a directory, assume the file is named "predictions.csv"
        potential_file = os.path.join(target_path, "predictions.csv")
        if os.path.exists(potential_file):
            target_path = potential_file
        else:
            print(f"[Error] Directory provided but 'predictions.csv' not found in: {target_path}")
            sys.exit(1)

    analyze_and_plot(target_path, gt_col=args.gt_col, threshold=args.threshold)