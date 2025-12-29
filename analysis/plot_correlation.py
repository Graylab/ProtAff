import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys
import argparse

# ================= CONFIGURATION (DEFAULTS) =================
# These act as defaults if not provided via arguments
DEFAULT_STRUCT_SCORES_CSV = "data/boltz2/adaptyv/all_models_scores.csv" 
DEFAULT_GT_CSV = "data/test/test_adaptyv.csv"                            

STRUCT_METRICS = ['ipSAE', 'ipTM_af', 'pDockQ', 'pDockQ2', 'LIS']
AGG_METHODS = ['max', 'min', 'mean', 'median']
# =================================================

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

    # Aggregation
    print(f"Aggregating {len(struct_df)} structural models...")
    # Filter only metrics that exist in the dataframe
    available_metrics = [m for m in STRUCT_METRICS if m in struct_df.columns]
    
    grouped_df = struct_df.groupby('id')[available_metrics].agg(AGG_METHODS)
    grouped_df.columns = ['_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()

    # Merging
    merged_df = pd.merge(grouped_df, gt_df[['id', 'log_Aff']], on='id', how='inner')
    merged_df = pd.merge(merged_df, aff_pred_df[['id', 'predicted_affinity']], on='id', how='inner')

    # --- TRANSFORMATION (Higher = Stronger) ---
    merged_df['neg_log_Aff'] = -1 * merged_df['log_Aff']
    merged_df['neg_predicted_affinity'] = -1 * merged_df['predicted_affinity']

    n_samples = len(merged_df)
    print(f"Merged {n_samples} targets.")
    
    if n_samples < 3:
        print("Not enough data points (>2 required).")
        return

    all_stats = []

    # Loop Aggregations
    for agg in AGG_METHODS:
        print(f"\n--- Processing Aggregation: {agg.upper()} ---")
        
        setup_plotting()
        fig, axes = plt.subplots(2, 3, figsize=(22, 16))
        axes = axes.flatten()
        
        # Define columns to plot
        plot_targets = []
        for m in available_metrics:
            plot_targets.append((f"{m}_{agg}", f"{m} ({agg})"))
        plot_targets.append(('neg_predicted_affinity', 'Predicted Affinity'))

        # Limit to available axes
        for i, (col_name, label) in enumerate(plot_targets):
            if i >= len(axes): break
            ax = axes[i]
            
            clean_data = merged_df[[col_name, 'neg_log_Aff']].dropna()
            if clean_data.empty: continue

            x = clean_data[col_name]
            y = clean_data['neg_log_Aff']

            # Calculate Stats
            pearson_r, p_val = stats.pearsonr(x, y)
            spearman_rho, s_p_val = stats.spearmanr(x, y)

            all_stats.append({
                'Aggregation': agg,
                'Metric': label,
                'Pearson_R': pearson_r,
                'Spearman_Rho': spearman_rho,
                'P_Value': p_val,
                'N': len(x)
            })

            # Plot
            sns.regplot(
                data=merged_df, 
                x=col_name, 
                y='neg_log_Aff', 
                ax=ax,
                scatter_kws={'alpha': 0.6, 'edgecolor': 'w', 's': 120}, # Larger dots
                line_kws={'color': '#d62728', 'alpha': 0.8, 'linewidth': 4} # Thicker line
            )

            # --- ANNOTATION BOX ---
            box_color = '#e6fffa' if pearson_r > 0 else '#ffe6e6'
            stats_text = (
                f"Pearson R = {pearson_r:.2f}\n"
                f"Spearman $\\rho$ = {spearman_rho:.2f}\n"
                f"P-value = {p_val:.1e}"
            )

            ax.text(0.05, 0.95, stats_text, 
                    transform=ax.transAxes, fontsize=16, 
                    verticalalignment='top', fontweight='medium',
                    bbox=dict(facecolor=box_color, alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

            # --- TITLES AND LABELS ---
            ax.set_title(label, fontsize=22, fontweight='bold', pad=15)
            
            if "Predicted Affinity" in label:
                xlabel = "Predicted -log_Aff"
            else:
                xlabel = f"Predicted {label}"
            
            ax.set_xlabel(xlabel, fontsize=18, fontweight='bold')
            ax.set_ylabel("Ground Truth -log_Aff", fontsize=18, fontweight='bold')
            
            ax.grid(True, linestyle='--', alpha=0.5)

        # Output
        out_img_path = os.path.join(output_dir, f"correlation_{agg}_final.png")
        fig.suptitle(f"Binder Selection Metrics ({agg.upper()} aggregation) | N={n_samples}", fontsize=26, fontweight='bold', y=0.99)
        
        plt.tight_layout()
        plt.savefig(out_img_path, dpi=300)
        print(f"Saved plot: {out_img_path}")
        plt.close(fig)

    # 5. Save Summary
    save_summary(all_stats, output_dir)

def save_summary(all_stats, output_dir):
    stats_df = pd.DataFrame(all_stats)
    out_csv_path = os.path.join(output_dir, "analysis_summary_full.csv")
    stats_df.to_csv(out_csv_path, index=False)
    
    # --- PLOTTING RANKING BARPLOT ---
    print("Generating Spearman ranking plot...")
    
    # 1. Deduplicate
    plot_df = stats_df.drop_duplicates(subset=['Metric', 'Spearman_Rho']).copy()
    
    # 2. Sort High to Low
    plot_df = plot_df.sort_values(by="Spearman_Rho", ascending=False)
    
    # 3. Setup Figure
    plt.figure(figsize=(14, max(8, len(plot_df) * 0.6)))
    sns.set_style("whitegrid")
    
    # 4. Create Horizontal Bar Plot
    ax = sns.barplot(
        data=plot_df,
        x="Spearman_Rho",
        y="Metric",
        palette="viridis",
        edgecolor="0.2"
    )
    
    # 5. Add Value Labels to Bars
    for p in ax.patches:
        width = p.get_width()
        if width >= 0:
            ha = 'left'; x_offset = 0.02
        else:
            ha = 'right'; x_offset = -0.02
            
        ax.text(
            width + x_offset,
            p.get_y() + p.get_height() / 2,
            f'{width:.3f}',
            ha=ha, va='center',
            fontsize=12, fontweight='bold', color='#333333'
        )

    # 6. Highlight "Predicted Affinity"
    for tick_label in ax.get_yticklabels():
        if "Predicted Affinity" in tick_label.get_text():
            tick_label.set_fontweight('bold')
            tick_label.set_fontsize(16)
        else:
            tick_label.set_fontweight('normal')

    # 7. Adjust X-Axis
    data_min = plot_df['Spearman_Rho'].min()
    data_max = plot_df['Spearman_Rho'].max()
    lower_bound = min(0, data_min); upper_bound = max(0, data_max)
    span = upper_bound - lower_bound
    if span == 0: span = 1.0
    ax.set_xlim(lower_bound - (span * 0.25), upper_bound + (span * 0.25))

    plt.axvline(0, color='black', linewidth=1.5, linestyle='-') 
    plt.title("Spearman Correlation Ranking (High to Low)", fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("Spearman Coefficient (Higher is Better)", fontsize=16, fontweight='bold')
    plt.ylabel("") 
    plt.xticks(fontsize=12)
    
    out_plot_path = os.path.join(output_dir, "spearman_ranking_barplot.png")
    plt.tight_layout()
    plt.savefig(out_plot_path, dpi=300)
    print(f"Saved ranking plot: {out_plot_path}")
    plt.close()

    # --- TEXT SUMMARY ---
    print("\n" + "="*50)
    print("RANKING: MOST IMPORTANT METRICS")
    print("="*50)
    ranked = stats_df.sort_values(by="Spearman_Rho", ascending=False)
    print(ranked[['Aggregation', 'Metric', 'Spearman_Rho', 'P_Value']].head(10).to_string(index=False))

def setup_plotting():
    """Sets visual styles for the plots with LARGE FONTS"""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 16,               
        'axes.titlesize': 22,         
        'axes.labelsize': 18,         
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.titlesize': 26
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Structural vs Predicted Correlations.")
    parser.add_argument("path", type=str, help="Path to predictions csv OR directory containing predictions.csv")
    parser.add_argument("--gt", type=str, default=DEFAULT_GT_CSV, help="Path to Ground Truth CSV")
    parser.add_argument("--struct", type=str, default=DEFAULT_STRUCT_SCORES_CSV, help="Path to Structural Scores CSV")
    
    args = parser.parse_args()
    
    target_path = args.path
    
    # Auto-infer logic
    if os.path.isdir(target_path):
        potential_file = os.path.join(target_path, "predictions.csv")
        if os.path.exists(potential_file):
            target_path = potential_file
        else:
            print(f"[Error] Directory provided but 'predictions.csv' not found in: {target_path}")
            sys.exit(1)
            
    # Derive output directory from the final file path
    output_dir = os.path.dirname(target_path)
    
    analyze_results(target_path, output_dir, args.gt, args.struct)