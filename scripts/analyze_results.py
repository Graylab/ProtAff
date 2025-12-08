import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, average_precision_score, r2_score

# --- CONFIGURATION ---
# Set to True for log(Kd) or dG (where -9 is better than -5)
# Set to False for pKd or Affinity Score (where 9 is better than 5)
LOWER_IS_BETTER = True 

def calculate_enrichment_factor(y_true, y_pred, top_percent, binary_threshold):
    """
    Calculates EF based on sorting order defined by LOWER_IS_BETTER.
    """
    n_total = len(y_true)
    n_top = int(n_total * (top_percent / 100.0))
    if n_top == 0: n_top = 1

    # 1. Identify Actives (True Binders)
    if LOWER_IS_BETTER:
        # Good binders have values LOWER than threshold (e.g. < -9)
        actives_mask = y_true <= binary_threshold
    else:
        # Good binders have values HIGHER than threshold (e.g. > 9)
        actives_mask = y_true >= binary_threshold

    total_actives = actives_mask.sum()
    background_rate = total_actives / n_total

    if total_actives == 0: return 0.0

    # 2. Sort Predictions
    if LOWER_IS_BETTER:
        # Sort Ascending (Lowest predicted values come first)
        sorted_indices = np.argsort(y_pred)
    else:
        # Sort Descending (Highest predicted values come first)
        sorted_indices = np.argsort(y_pred)[::-1]

    # 3. Select Top K
    top_indices = sorted_indices[:n_top]
    
    # 4. Count Hits
    hits_in_top = actives_mask[top_indices].sum()
    
    # 5. Calculate EF
    selection_rate = hits_in_top / n_top
    return selection_rate / background_rate

def analyze_and_plot(csv_path):
    # 1. Load Data
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if "log_Aff" not in df.columns or "predicted_affinity" not in df.columns:
        print(f"[Error] CSV must contain 'log_Aff' and 'predicted_affinity'.")
        return

    y_true = df["log_Aff"].values
    y_pred = df["predicted_affinity"].values
    
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
    
    # Define "Good Binder" threshold based on Ground Truth distribution
    # If Lower is Better, the best binders are in the BOTTOM 10th percentile
    if LOWER_IS_BETTER:
        binary_threshold = np.percentile(y_true, 10) 
        print(f"[Config] Lower is Better. 'Good' defined as log_Aff <= {binary_threshold:.2f}")
        # Binary Labels (1 = Good, 0 = Bad)
        y_bin = (y_true <= binary_threshold).astype(int)
        # For AUC calculation, we need to invert the predictions 
        # (because sklearn assumes Higher Prediction = Class 1)
        y_score_for_auc = -y_pred 
    else:
        binary_threshold = np.percentile(y_true, 90)
        print(f"[Config] Higher is Better. 'Good' defined as log_Aff >= {binary_threshold:.2f}")
        y_bin = (y_true >= binary_threshold).astype(int)
        y_score_for_auc = y_pred

    # AUC Scores
    if len(np.unique(y_bin)) > 1:
        auroc = roc_auc_score(y_bin, y_score_for_auc)
        auprc = average_precision_score(y_bin, y_score_for_auc)
    else:
        auroc, auprc = 0.5, 0.0

    # Enrichment Factors
    ef_1 = calculate_enrichment_factor(y_true, y_pred, 1.0, binary_threshold)
    ef_5 = calculate_enrichment_factor(y_true, y_pred, 5.0, binary_threshold)
    ef_10 = calculate_enrichment_factor(y_true, y_pred, 10.0, binary_threshold)

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
    print(f"\n--- Screening Power (Top 10% Target) ---")
    print(f"AUROC        : {auroc:.4f}")
    print(f"AUPRC        : {auprc:.4f}")
    print(f"EF @ 1%      : {ef_1:.2f}x")
    print(f"EF @ 5%      : {ef_5:.2f}x")
    print(f"EF @ 10%     : {ef_10:.2f}x")
    print("-" * 40)

    # Save Metrics
    txt_path = os.path.join(output_dir, f"{base_name}_metrics.txt")
    with open(txt_path, "w") as f:
        f.write(f"Pearson R: {pearson_r:.4f}\n")
        f.write(f"Spearman Rho: {spearman_rho:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"AUROC: {auroc:.4f}\n")
        f.write(f"EF 1%: {ef_1:.4f}\n")
        f.write(f"EF 10%: {ef_10:.4f}\n")

    # -----------------------------------------------------------
    # 4. Plot (NEGATED values)
    # -----------------------------------------------------------
    sns.set_theme(style="ticks", context="talk", palette="deep")
    
    # NEGATING both axes as requested
    x_plot = -y_pred
    y_plot = -y_true

    g = sns.JointGrid(x=x_plot, y=y_plot, height=8, ratio=5)

    g.plot_joint(sns.regplot, 
                 color="#4C72B0",
                 truncate=False,
                 scatter_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'white', 'linewidths': 0.8, 'zorder': 2},
                 line_kws={'color': "#C44E52", 'alpha': 0.9, 'linewidth': 2.5, 'label': 'Linear Fit', 'zorder': 3})

    # Identity Line
    min_val = min(y_plot.min(), x_plot.min())
    max_val = max(y_plot.max(), x_plot.max())
    lims = [min_val, max_val]
    #g.ax_joint.plot(lims, lims, color="#333333", linestyle="--", linewidth=1.5, alpha=0.6, label="Ideal (y=x)", zorder=1)

    g.plot_marginals(sns.kdeplot, color="#4C72B0", fill=True, alpha=0.3)

    # UPDATED Labels to reflect negation
    g.set_axis_labels(xlabel="-Predicted Affinity", ylabel="-Ground Truth (log_Aff)")
    g.ax_joint.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='gray')

    stats_text = (
        f"Spearman: {spearman_rho:.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"EF @ 10%: {ef_10:.1f}x\n"
        f"AUC: {auroc:.2f}"
    )
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='#d3d3d3')
    g.ax_joint.text(0.05, 0.95, stats_text,
                    transform=g.ax_joint.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=props, fontsize=12, zorder=5)

    g.ax_joint.legend(loc='lower right', frameon=True, framealpha=0.9)

    plot_path = os.path.join(output_dir, f"{base_name}_scatter.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"[Success] Plot saved to: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to prediction CSV")
    args = parser.parse_args()
    analyze_and_plot(args.csv)