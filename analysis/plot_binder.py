import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
import os
import argparse
import numpy as np

# ================= CONFIGURATION =================
GT_CSV = "data/binder/test_design.csv" 

METRICS_CONFIG = {
    'AF3 ipSAE': ('af3_ipSAE_min', False),    # Higher is better
    'Predicted Affinity': ('predicted_affinity', True), # Lower is better
}

PLOT_SETTINGS = {
    'figsize_width': 12,
    'bar_height_per_target': 0.6,
    'font_scale': 1.2
}
# =================================================

def get_ap(y_true, y_scores):
    mask = y_true.notna() & y_scores.notna()
    y_t, y_s = y_true[mask], y_scores[mask]
    if len(y_t) == 0 or y_t.nunique() < 2: return 0.0
    return average_precision_score(y_t, y_s)

def get_auroc(y_true, y_scores):
    mask = y_true.notna() & y_scores.notna()
    y_t, y_s = y_true[mask], y_scores[mask]
    if len(y_t) == 0 or y_t.nunique() < 2: return 0.5 # Random baseline
    return roc_auc_score(y_t, y_s)

def print_metrics_summary(plot_df):
    """Prints detailed per-target metrics and the average across targets."""
    
    # 1. Detailed Per-Target Table
    print("\n" + "="*100)
    print(f"{'DETAILED PER-TARGET METRICS':^100}")
    print("="*100)
    # Reorganizing for display
    display_df = plot_df.copy()
    display_df = display_df.sort_values(['Method', 'target_id'])
    print(display_df.to_string(index=False, formatters={
        'AP': '{:,.3f}'.format, 
        'AUROC': '{:,.3f}'.format,
        'Success@1': '{:,.0f}'.format,
        'Success@10': '{:,.0f}'.format
    }))

    # 2. Average (Global) Summary Table
    print("\n" + "="*100)
    print(f"{'AVERAGE METRICS ACROSS ALL TARGETS':^100}")
    print("="*100)
    summary = plot_df.groupby('Method').agg({
        'AP': 'mean',
        'AUROC': 'mean',
        'Success@1': 'mean',
        'Success@10': 'mean'
    }).reset_index()
    
    # Rename columns for clarity in terminal
    summary.columns = ['Method', 'Mean AP', 'Mean AUROC', 'Success Rate @1', 'Success Rate @10']
    print(summary.to_string(index=False, formatters={
        'Mean AP': '{:,.3f}'.format,
        'Mean AUROC': '{:,.3f}'.format,
        'Success Rate @1': '{:,.2%}'.format,
        'Success Rate @10': '{:,.2%}'.format
    }))
    print("="*100 + "\n")

def analyze_results(pred_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    gt_df = pd.read_csv(GT_CSV)
    pred_df = pd.read_csv(pred_csv)

    gt_df['id'] = gt_df['id'].astype(str).str.strip()
    pred_df['id'] = pred_df['id'].astype(str).str.strip()

    cols_to_merge = ['id'] + [v[0] for v in METRICS_CONFIG.values() if v[0] in pred_df.columns]
    merged_df = gt_df.merge(pred_df[cols_to_merge], on='id', how='left').dropna(subset=['target_id'])

    targets = sorted(merged_df['target_id'].unique())
    plot_data_list = []

    for label, (col, lower_is_better) in METRICS_CONFIG.items():
        if col not in merged_df.columns: continue
        
        for t in targets:
            group = merged_df[merged_df['target_id'] == t].copy()
            # Ensure numeric
            group['is_binder'] = pd.to_numeric(group['is_binder'], errors='coerce').fillna(0)
            group[col] = pd.to_numeric(group[col], errors='coerce')
            
            # For AP/AUROC, we need "higher is better" scores
            scores = -group[col] if lower_is_better else group[col]
            
            # Ranking metrics
            ap_score = get_ap(group['is_binder'], scores)
            auroc_score = get_auroc(group['is_binder'], scores)
            
            # Success metrics (Binary: Did we find at least one binder?)
            group_sorted = group.sort_values(by=col, ascending=lower_is_better)
            s1 = 1 if group_sorted.head(1)['is_binder'].sum() >= 1 else 0
            s10 = 1 if group_sorted.head(10)['is_binder'].sum() >= 1 else 0
            
            plot_data_list.append({
                'target_id': t, 
                'Method': label,
                'AP': ap_score, 
                'AUROC': auroc_score, 
                'Success@1': s1, 
                'Success@10': s10
            })

    plot_df = pd.DataFrame(plot_data_list)

    # 1. Print tables to terminal
    print_metrics_summary(plot_df)
    
    # 2. Generate Per-Target Plots
    for m in ['AP', 'AUROC', 'Success@1', 'Success@10']:
        generate_per_target_plot(plot_df, output_dir, metric=m)
    
    # 3. Generate Average/Global Summary Plot
    plot_global_summary(plot_df, output_dir)

def generate_per_target_plot(df, output_dir, metric):
    sns.set_context("paper", font_scale=PLOT_SETTINGS['font_scale'])
    h = max(6, len(df['target_id'].unique()) * PLOT_SETTINGS['bar_height_per_target'])
    plt.figure(figsize=(PLOT_SETTINGS['figsize_width'], h))
    
    ax = sns.barplot(data=df, x=metric, y='target_id', hue='Method', edgecolor='black')
    
    # Formatting
    fmt = '%.2f' if metric in ['AP', 'AUROC'] else '%d'
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=3)
    
    plt.title(f'Per-Target {metric}', weight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"per_target_{metric.lower().replace('@','_')}.png"), dpi=300)
    plt.close()

def plot_global_summary(plot_df, output_dir):
    """Plots the average of all metrics across targets."""
    summary_df = plot_df.groupby('Method')[['AP', 'AUROC', 'Success@1', 'Success@10']].mean().reset_index()
    melted = summary_df.melt(id_vars='Method', var_name='Metric', value_name='Average Value')
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=melted, x='Metric', y='Average Value', hue='Method', palette='magma')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
        
    plt.title('Average Metrics Across All Targets', weight='bold')
    plt.ylim(0, 1.1)
    plt.ylabel('Score / Success Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_summary_metrics.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to predictions.csv or directory containing it")
    args = parser.parse_args()
    
    target_path = args.path if not os.path.isdir(args.path) else os.path.join(args.path, "predictions.csv")
    analyze_results(target_path, os.path.dirname(target_path))