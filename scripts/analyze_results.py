"""
scripts/analyze_results.py
Analysis: Metrics + Professional Scatter Plot with KDE, Fit Line, and y=x Identity Line.

Usage:
    python scripts/analyze_results.py --csv "inference_results/test_data/predictions.csv"
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    
    # Output setup
    output_dir = os.path.dirname(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    # 2. Calculate Metrics
    pearson_r, _ = stats.pearsonr(y_true, y_pred)
    spearman_rho, _ = stats.spearmanr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Console Output
    print("-" * 30)
    print(f"RESULTS: {os.path.basename(csv_path)}")
    print("-" * 30)
    print(f"Pearson R    : {pearson_r:.4f}")
    print(f"Spearman Rho : {spearman_rho:.4f}")
    print(f"RMSE         : {rmse:.4f}")
    print(f"MAE          : {mae:.4f}")
    print("-" * 30)

    # Save Metrics
    txt_path = os.path.join(output_dir, f"{base_name}_metrics.txt")
    with open(txt_path, "w") as f:
        f.write(f"Pearson R: {pearson_r:.4f}\n")
        f.write(f"Spearman Rho: {spearman_rho:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")

    # 3. Plot Generation
    sns.set_theme(style="ticks", context="talk", palette="deep")
    
    # Define colors for a professional look
    scatter_color = "#4C72B0"  # Deep Blue
    line_color = "#C44E52"     # Muted Red
    identity_color = "#333333" # Dark Grey
    
    # Initialize JointGrid with specific ratio and size
    g = sns.JointGrid(x=y_true, y=y_pred, height=8, ratio=5)

    # A. Plot Joint Area (Scatter + Fit Line)
    g.plot_joint(sns.regplot, 
                 color=scatter_color,
                 truncate=False,
                 scatter_kws={
                     'alpha': 0.6, 
                     's': 80, 
                     'edgecolor': 'white', 
                     'linewidths': 0.8,
                     'zorder': 2  # Make sure dots are on top of y=x line
                 },
                 line_kws={
                     'color': line_color, 
                     'alpha': 0.9, 
                     'linewidth': 2.5, 
                     'label': 'Linear Fit',
                     'zorder': 3
                 })

    # --- ADDED: y=x Identity Line ---
    # Calculate min/max range to ensure line covers the whole plot
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    padding = (max_val - min_val) * 0.05 # 5% padding
    
    lims = [min_val - padding, max_val + padding]
    
    g.ax_joint.plot(lims, lims, 
                    color=identity_color, 
                    linestyle="--", 
                    linewidth=1.5, 
                    alpha=0.6, 
                    label="Ideal (y=x)", 
                    zorder=1) # zorder 1 puts it behind the scatter points

    # B. Plot Margins (KDE)
    g.plot_marginals(sns.kdeplot, color=scatter_color, fill=True, alpha=0.3, linewidth=1.5)

    # C. Aesthetics: Labels and Grid
    g.set_axis_labels(xlabel="Ground Truth (log_Aff)", ylabel="Predicted Affinity")
    g.ax_joint.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='gray')

    # D. Add Metrics Text Box
    stats_text = (
        f"Pearson R: {pearson_r:.2f}\n"
        f"Spearman: {spearman_rho:.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"MAE: {mae:.2f}"
    )
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='#d3d3d3')
    g.ax_joint.text(0.05, 0.95, stats_text,
                    transform=g.ax_joint.transAxes,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=props,
                    fontsize=13,
                    zorder=5)

    # E. Add Legend (now includes both Linear Fit and y=x)
    g.ax_joint.legend(loc='lower right', frameon=True, framealpha=0.9)

    # Save Plot
    plot_path = os.path.join(output_dir, f"{base_name}_scatter.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"[Success] Plot saved to: {plot_path}")
    print(f"[Success] Metrics saved to: {txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to prediction CSV")
    args = parser.parse_args()
    
    analyze_and_plot(args.csv)