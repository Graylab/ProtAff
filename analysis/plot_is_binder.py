import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve
import os

# ================= CONFIGURATION =================
GT_CSV = "data/binder_design/test_design.csv"                   # Ground Truth
PRED_CSV = "inference_output/base_from_pretrain_3/test_binder/predictions.csv"
OUTPUT_DIR = "analysis_output/ap_plots"

METRICS_CONFIG = {
    'AF3 ipSAE': ('af3_ipSAE_min', False),                 # Higher is better
    'Predicted Affinity': ('predicted_affinity', True),    # Lower is better (negated)
    #'Predicted Prob Binder': ('predicted_prob_binder', False) # Higher is better
}

PLOT_SETTINGS = {
    'figsize_width': 12,
    'bar_height_per_target': 1.0,
    'font_scale': 1.4
}
# =================================================

def get_ap(y_true, y_scores):
    """Helper function to calculate AP safely"""
    y_true = pd.to_numeric(y_true, errors='coerce')
    y_scores = pd.to_numeric(y_scores, errors='coerce')
    
    mask = y_true.notna() & y_scores.notna()
    y_true_clean = y_true[mask]
    y_scores_clean = y_scores[mask]
    
    if y_true_clean.nunique() < 2:
        return 0.0
    return average_precision_score(y_true_clean, y_scores_clean)

# --- NEW FUNCTION: Plot PR Curves ---
def plot_global_pr_curve(merged_df, metrics_config, output_dir):
    """Plots a single figure with PR curves for all metrics (Global Pool)."""
    plt.figure(figsize=(10, 8))
    
    for label, (col, lower_is_better) in metrics_config.items():
        if col not in merged_df.columns:
            continue

        # Prepare Data
        y_true = pd.to_numeric(merged_df['is_binder'], errors='coerce')
        y_scores = pd.to_numeric(merged_df[col], errors='coerce')

        # Handle directionality (Critical for Affinity)
        if lower_is_better:
            y_scores = -y_scores 

        # Clean NaNs
        mask = y_true.notna() & y_scores.notna()
        y_true_clean = y_true[mask]
        y_scores_clean = y_scores[mask]

        if y_true_clean.nunique() < 2:
            print(f"[PR Plot] Skipping {label}: Not enough class diversity.")
            continue

        # Calculate Curve
        precision, recall, _ = precision_recall_curve(y_true_clean, y_scores_clean)
        ap_score = average_precision_score(y_true_clean, y_scores_clean)

        # Plot
        plt.plot(recall, precision, lw=2.5, label=f'{label} (AP = {ap_score:.2f})')

    # Styling
    plt.xlabel('Recall (Sensitivity)', fontsize=14, fontweight='bold')
    plt.ylabel('Precision (PPV)', fontsize=14, fontweight='bold')
    plt.title('Global Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])

    # Save
    out_path = os.path.join(output_dir, "global_pr_curve.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved Global PR Curve: {out_path}")
    plt.show()

def analyze_results():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading data...")
    try:
        gt_df = pd.read_csv(GT_CSV)
        pred_df = pd.read_csv(PRED_CSV)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Cleanup IDs
    gt_df['id'] = gt_df['id'].astype(str).str.strip()
    pred_df['id'] = pred_df['id'].astype(str).str.strip()

    print(f"Merging Ground Truth ({len(gt_df)}) with Predictions ({len(pred_df)})...")
    cols_to_merge = ['id'] + [v[0] for v in METRICS_CONFIG.values() if v[0] in pred_df.columns]
    merged_df = gt_df.merge(pred_df[cols_to_merge], on='id', how='left')
    print(f"Merged {len(merged_df)} samples.")

    # --- 1. GENERATE PR CURVES (New Step) ---
    plot_global_pr_curve(merged_df, METRICS_CONFIG, OUTPUT_DIR)

    # --- 2. EXISTING AP BAR CHART LOGIC ---
    plot_data_list = []
    targets = merged_df['target_id'].unique()
    
    print(f"Calculating AP bar stats for {len(targets)} targets...")
    for t in targets:
        group = merged_df[merged_df['target_id'] == t]
        for label, (col, lower_is_better) in METRICS_CONFIG.items():
            if col not in group.columns: continue
            
            scores = group[col]
            if lower_is_better: scores = -scores
            
            score = get_ap(group['is_binder'], scores)
            plot_data_list.append({'target_id': t, 'AP': score, 'Method': label})

    plot_df = pd.DataFrame(plot_data_list)
    
    # Aggregates
    aggs = []
    # Mean AP
    if not plot_df.empty:
        mean_scores = plot_df.groupby('Method')['AP'].mean()
        for label, score in mean_scores.items():
            aggs.append({'target_id': 'Average (Mean)', 'AP': score, 'Method': label})

    # Global Pooled AP
    for label, (col, lower_is_better) in METRICS_CONFIG.items():
        if col not in merged_df.columns: continue
        scores_all = merged_df[col]
        if lower_is_better: scores_all = -scores_all
        global_score = get_ap(merged_df['is_binder'], scores_all)
        aggs.append({'target_id': 'Global (Pooled)', 'AP': global_score, 'Method': label})

    # Sort
    first_metric_label = list(METRICS_CONFIG.keys())[0]
    sort_subset = plot_df[plot_df['Method'] == first_metric_label]
    if not sort_subset.empty:
        target_order = sort_subset.sort_values('AP', ascending=False)['target_id']
        sorted_plot_df = plot_df.set_index('target_id').loc[target_order].reset_index()
    else:
        sorted_plot_df = plot_df

    final_df = pd.concat([sorted_plot_df, pd.DataFrame(aggs)], ignore_index=True)

    save_summary(final_df)
    generate_plot(final_df)

def save_summary(df):
    out_csv_path = os.path.join(OUTPUT_DIR, "ap_summary_full.csv")
    df.to_csv(out_csv_path, index=False)
    print(f"Saved summary CSV: {out_csv_path}")

def generate_plot(final_df):
    setup_plotting()
    num_targets = len(final_df['target_id'].unique())
    fig_height = max(6, num_targets * PLOT_SETTINGS['bar_height_per_target'] + 2) # Ensure min height
    
    plt.figure(figsize=(PLOT_SETTINGS['figsize_width'], fig_height))
    ax = sns.barplot(data=final_df, x='AP', y='target_id', hue='Method', 
                     edgecolor='black', palette='muted')
    
    plt.title('Average Precision Comparison', weight='bold', pad=20)
    plt.xlabel('Average Precision (AP)', weight='bold')
    plt.ylabel('') 
    plt.xlim(0, 1.1)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=11)

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Metric')
    plt.tight_layout()

    out_plot_path = os.path.join(OUTPUT_DIR, 'ap_comparison_final.png')
    plt.savefig(out_plot_path, dpi=300)
    print(f"Saved Bar Plot: {out_plot_path}")
    plt.show()

def setup_plotting():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=PLOT_SETTINGS['font_scale'])
    plt.rcParams.update({'font.family': 'sans-serif'})

if __name__ == "__main__":
    analyze_results()
