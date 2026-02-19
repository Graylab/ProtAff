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

def setup_slide_style():
    """Sets visual styles sized for presentation slides."""
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
        setup_slide_style()
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

            is_ours = "Predicted Affinity" in label
            dot_color = '#D55E00' if is_ours else '#4C72B0'
            line_color = '#d62728' if is_ours else '#1f77b4'

            sns.regplot(data=merged_df, x=col_name, y='neg_log_Aff', ax=ax,
                        scatter_kws={'alpha': 0.6, 'edgecolor': 'w', 's': 120, 'color': dot_color},
                        line_kws={'color': line_color, 'alpha': 0.8, 'linewidth': 4})

            if is_ours:
                for spine in ax.spines.values():
                    spine.set_edgecolor('#D55E00')
                    spine.set_linewidth(3)

            box_color = '#e6fffa' if pearson_r > 0 else '#ffe6e6'
            stats_text = f"Pearson R = {pearson_r:.3f}\nSpearman $\\rho$ = {spearman_rho:.3f}\nP-value = {p_val:.1e}"
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=16,
                    verticalalignment='top', fontweight='medium',
                    bbox=dict(facecolor=box_color, alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

            ax.set_title(label, fontsize=22, fontweight='bold', pad=15,
                         color='#D55E00' if is_ours else 'black')
            ax.set_xlabel("-Predicted Affinity" if is_ours else label)
            ax.set_ylabel("-Ground Truth log_Aff")

        out_img_path = os.path.join(output_dir, f"correlation_{agg}_final.png")
        fig.suptitle(f"Binder Selection Metrics ({agg.upper()}) | N={n_samples}", fontsize=26, fontweight='bold', y=0.99)
        plt.tight_layout()
        plt.savefig(out_img_path, dpi=300)
        plt.close(fig)

    # 2. ADDED: FOCUSED 1x2 SUBPLOT (ipSAE_min vs Predicted_Affinity)
    print("\n[Analysis] Generating focused 1x2 ipSAE_min vs Predicted Affinity comparison...")
    ipsae_min_col = "ipSAE_min"
    if ipsae_min_col in merged_df.columns:
        setup_slide_style()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 20))

        comparison_tasks = [
            (ipsae_min_col, "ipSAE (min)", ax1, False),
            ('neg_predicted_affinity', "Predicted Affinity", ax2, True)
        ]

        for col, title, ax, is_ours in comparison_tasks:
            clean_data = merged_df[[col, 'neg_log_Aff']].dropna()
            if clean_data.empty: continue

            x, y = clean_data[col], clean_data['neg_log_Aff']
            r, p = stats.pearsonr(x, y)
            rho, _ = stats.spearmanr(x, y)

            dot_color = '#D55E00' if is_ours else '#0072B2'
            line_color = '#d62728' if is_ours else '#1f77b4'

            sns.regplot(data=clean_data, x=col, y='neg_log_Aff', ax=ax,
                        scatter_kws={'alpha': 0.6, 's': 150, 'edgecolor': 'w', 'color': dot_color},
                        line_kws={'color': line_color, 'linewidth': 4})

            if is_ours:
                for spine in ax.spines.values():
                    spine.set_edgecolor('#D55E00')
                    spine.set_linewidth(3)

            ax.text(0.05, 0.95, f"Pearson R: {r:.3f}\nSpearman $\\rho$: {rho:.3f}\nP: {p:.1e}",
                    transform=ax.transAxes, fontsize=16, verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5', edgecolor='gray'))

            ax.set_title(title, fontsize=22, fontweight='bold', pad=20,
                         color='#D55E00' if is_ours else 'black')
            ax.set_ylabel("-Ground Truth log_Aff")
            ax.set_xlabel(f"-Predicted Affinity" if is_ours else col)
            ax.grid(True, linestyle='--', alpha=0.3)

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
    setup_slide_style()
    plot_df = stats_df.drop_duplicates(subset=['Metric', 'Spearman_Rho']).copy()
    plot_df = plot_df.sort_values(by="Spearman_Rho", ascending=False)

    # Build color list: highlight Predicted Affinity
    bar_colors = ['#D55E00' if 'Predicted Affinity' in m else '#0072B2'
                  for m in plot_df['Metric']]

    plt.figure(figsize=(14, max(8, len(plot_df) * 0.6)))
    ax = sns.barplot(data=plot_df, x="Spearman_Rho", y="Metric", palette=bar_colors, edgecolor="0.2")

    for p in ax.patches:
        width = p.get_width()
        ha = 'left' if width >= 0 else 'right'
        x_offset = 0.02 if width >= 0 else -0.02
        ax.text(width + x_offset, p.get_y() + p.get_height() / 2, f'{width:.3f}',
                ha=ha, va='center', fontsize=14, fontweight='bold')

    # Highlight y-tick label for Predicted Affinity
    for tick_label in ax.get_yticklabels():
        if 'Predicted Affinity' in tick_label.get_text():
            tick_label.set_fontweight('bold')
            tick_label.set_color('#D55E00')

    plt.axvline(0, color='black', linewidth=1.5)
    plt.xlim(-0.4, 0.4)
    plt.ylabel("")
    plt.title("Spearman Correlation Ranking", fontweight='bold', pad=20)
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