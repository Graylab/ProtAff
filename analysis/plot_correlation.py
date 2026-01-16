import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys
import argparse

# ================= CONFIGURATION (DEFAULTS) =================
DEFAULT_STRUCT_SCORES_CSV = "data/boltz2/adaptyv/all_models_scores.csv" 
DEFAULT_GT_CSV = "data/test/test_adaptyv.csv"                       

STRUCT_METRICS = ['ipSAE', 'ipTM_af', 'pDockQ', 'pDockQ2', 'LIS']
AGG_METHODS = ['max', 'min', 'mean', 'median']
# ============================================================

def setup_plotting():
    """Sets visual styles for the plots with LARGE FONTS"""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 20,                
        'axes.titlesize': 22,         
        'axes.labelsize': 18,         
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.titlesize': 26
    })

def analyze_results(pred_csv, output_dir, gt_csv, struct_csv):
    print(f"\n[Analysis] Predictions: {pred_csv}")
    print(f"[Analysis] Output Dir : {output_dir}")
    print(f"[Analysis] Ground Truth: {gt_csv}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading data...")
    try:
        struct_df = pd.read_csv(struct_csv)
        gt_df = pd.read_csv(gt_csv)
        aff_pred_df = pd.read_csv(pred_csv)
    except FileNotFoundError as e:
        print(f"[Error] File not found: {e}")
        return

    # Cleanup IDs
    struct_df['id'] = struct_df['id'].astype(str).str.strip()
    gt_df['id'] = gt_df['id'].astype(str).str.strip()
    aff_pred_df['id'] = aff_pred_df['id'].astype(str).str.strip()

    # Aggregation of Structural Metrics
    print(f"Aggregating {len(struct_df)} structural models...")
    available_metrics = [m for m in STRUCT_METRICS if m in struct_df.columns]
    
    grouped_df = struct_df.groupby('id')[available_metrics].agg(AGG_METHODS)
    grouped_df.columns = ['_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()

    # Merging Dataframes
    merged_df = pd.merge(grouped_df, gt_df[['id', 'log_Aff']], on='id', how='inner')
    merged_df = pd.merge(merged_df, aff_pred_df[['id', 'predicted_affinity']], on='id', how='inner')

    # Transformation
    merged_df['neg_log_Aff'] = -1 * merged_df['log_Aff']
    merged_df['neg_predicted_affinity'] = -1 * merged_df['predicted_affinity']

    n_samples = len(merged_df)
    print(f"Merged {n_samples} targets.")
    
    if n_samples < 3:
        print("Not enough data points (>2 required).")
        return

    all_stats = []

    # 1. MAIN ANALYSIS LOOP (Original Grids)
    for agg in AGG_METHODS:
        print(f"--- Processing Aggregation: {agg.upper()} ---")
        setup_plotting()
        fig, axes = plt.subplots(2, 3, figsize=(22, 16))
        axes = axes.flatten()
        
        plot_targets = []
        for m in available_metrics:
            plot_targets.append((f"{m}_{agg}", f"{m} ({agg})"))
        plot_targets.append(('neg_predicted_affinity', 'Predicted Affinity'))

        for i, (col_name, label) in enumerate(plot_targets):
            if i >= len(axes): break
            ax = axes[i]
            clean_data = merged_df[[col_name, 'neg_log_Aff']].dropna()
            if clean_data.empty: continue

            x, y = clean_data[col_name], clean_data['neg_log_Aff']
            pearson_r, p_val = stats.pearsonr(x, y)
            spearman_rho, _ = stats.spearmanr(x, y)

            all_stats.append({
                'Aggregation': agg,
                'Metric': label,
                'Pearson_R': pearson_r,
                'Spearman_Rho': spearman_rho,
                'P_Value': p_val,
                'N': len(x)
            })

            sns.regplot(data=merged_df, x=col_name, y='neg_log_Aff', ax=ax,
                        scatter_kws={'alpha': 0.6, 'edgecolor': 'w', 's': 120},
                        line_kws={'color': '#d62728', 'alpha': 0.8, 'linewidth': 4})

            box_color = '#e6fffa' if pearson_r > 0 else '#ffe6e6'
            stats_text = f"Pearson R = {pearson_r:.3f}\nSpearman $\\rho$ = {spearman_rho:.3f}\nP-value = {p_val:.1e}"
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=16, 
                    verticalalignment='top', fontweight='medium',
                    bbox=dict(facecolor=box_color, alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

            ax.set_title(label, fontsize=22, fontweight='bold', pad=15)
            ax.set_xlabel("Predicted -log_Aff" if "Predicted Affinity" in label else f"Predicted {label}", fontsize=18)
            ax.set_ylabel("Ground Truth -log_Aff", fontsize=18)

        out_img_path = os.path.join(output_dir, f"correlation_{agg}_final.png")
        fig.suptitle(f"Binder Selection Metrics ({agg.upper()}) | N={n_samples}", fontsize=26, fontweight='bold', y=0.99)
        plt.tight_layout()
        plt.savefig(out_img_path, dpi=300)
        plt.close(fig)

    # 2. ADDED: FOCUSED 1x2 SUBPLOT (ipSAE_min vs Predicted_Affinity)
    print("\n[Analysis] Generating focused 1x2 ipSAE_min vs Predicted Affinity comparison...")
    ipsae_min_col = "ipSAE_min"
    if ipsae_min_col in merged_df.columns:
        setup_plotting()
        # Changed to 1 row, 2 columns. Swapped figsize to be wide (20x10)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 20))
        
        comparison_tasks = [
            (ipsae_min_col, "Structural: ipSAE (min) vs GT", ax1),
            ('neg_predicted_affinity', "Sequence: Predicted Affinity vs GT", ax2)
        ]
        
        for col, title, ax in comparison_tasks:
            clean_data = merged_df[[col, 'neg_log_Aff']].dropna()
            if clean_data.empty: continue
            
            x, y = clean_data[col], clean_data['neg_log_Aff']
            r, p = stats.pearsonr(x, y)
            rho, _ = stats.spearmanr(x, y)
            
            # Regression Plot
            sns.regplot(data=clean_data, x=col, y='neg_log_Aff', ax=ax,
                        scatter_kws={'alpha': 0.6, 's': 150, 'edgecolor': 'w'},
                        line_kws={'color': '#1f77b4' if 'ipSAE' in col else '#d62728', 'linewidth': 4})
            
            # Annotate with a slightly smaller font to fit horizontal layout
            ax.text(0.05, 0.95, f"Pearson R: {r:.3f}\nSpearman $\\rho$: {rho:.3f}\nP: {p:.1e}", 
                    transform=ax.transAxes, fontsize=16, verticalalignment='top', 
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5', edgecolor='gray'))
            
            ax.set_title(title, fontsize=22, fontweight='bold', pad=20)
            ax.set_ylabel("Ground Truth -log_Aff", fontsize=18)
            ax.set_xlabel(f"Predictor Score ({col})", fontsize=18)
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        focused_out = os.path.join(output_dir, "focused_ipsae_min_vs_predicted.png")
        plt.savefig(focused_out, dpi=300)
        print(f"Saved focused 1x2 comparison: {focused_out}")
        plt.close(fig)

    # 3. SAVE SUMMARY AND RANKING PLOT
    save_summary(all_stats, output_dir)

def save_summary(all_stats, output_dir):
    stats_df = pd.DataFrame(all_stats)
    out_csv_path = os.path.join(output_dir, "analysis_summary_full.csv")
    stats_df.to_csv(out_csv_path, index=False)
    
    print("Generating Spearman ranking plot...")
    plot_df = stats_df.drop_duplicates(subset=['Metric', 'Spearman_Rho']).copy()
    plot_df = plot_df.sort_values(by="Spearman_Rho", ascending=False)
    
    plt.figure(figsize=(14, max(8, len(plot_df) * 0.6)))
    ax = sns.barplot(data=plot_df, x="Spearman_Rho", y="Metric", palette="viridis", edgecolor="0.2")
    
    for p in ax.patches:
        width = p.get_width()
        ha = 'left' if width >= 0 else 'right'
        x_offset = 0.02 if width >= 0 else -0.02
        ax.text(width + x_offset, p.get_y() + p.get_height() / 2, f'{width:.3f}', 
                ha=ha, va='center', fontsize=12, fontweight='bold')

    plt.axvline(0, color='black', linewidth=1.5)
    plt.xlim(-0.4, 0.4)
    plt.title("Spearman Correlation Ranking (High to Low)", fontsize=20, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spearman_ranking_barplot.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Structural vs Predicted Correlations.")
    parser.add_argument("path", type=str, help="Path to predictions csv")
    parser.add_argument("--gt", type=str, default=DEFAULT_GT_CSV)
    parser.add_argument("--struct", type=str, default=DEFAULT_STRUCT_SCORES_CSV)
    
    args = parser.parse_args()
    target_path = args.path
    if os.path.isdir(target_path):
        target_path = os.path.join(target_path, "predictions.csv")
            
    analyze_results(target_path, os.path.dirname(target_path), args.gt, args.struct)