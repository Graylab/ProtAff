import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
import os
import sys
import argparse
import numpy as np

# ================= CONFIGURATION =================
GT_CSV = "data/binder/test_design.csv" 

METRICS_CONFIG = {
    'AF3 ipSAE': ('af3_ipSAE_min', False),    # Higher is better
    'Predicted Affinity': ('predicted_affinity', True), # Lower is better (negated)
}

PLOT_SETTINGS = {
    'figsize_width': 12,
    'bar_height_per_target': 0.8,
    'font_scale': 1.4
}
# =================================================

def get_ap(y_true, y_scores):
    y_true = pd.to_numeric(y_true, errors='coerce')
    y_scores = pd.to_numeric(y_scores, errors='coerce')
    mask = y_true.notna() & y_scores.notna()
    y_true_clean, y_scores_clean = y_true[mask], y_scores[mask]
    if len(y_true_clean) == 0 or y_true_clean.nunique() < 2: return 0.0
    return average_precision_score(y_true_clean, y_scores_clean)

def get_auroc(y_true, y_scores):
    y_true = pd.to_numeric(y_true, errors='coerce')
    y_scores = pd.to_numeric(y_scores, errors='coerce')
    mask = y_true.notna() & y_scores.notna()
    y_true_clean, y_scores_clean = y_true[mask], y_scores[mask]
    if len(y_true_clean) == 0 or y_true_clean.nunique() < 2: return 0.0
    return roc_auc_score(y_true_clean, y_scores_clean)

def print_terminal_metrics(plot_df, success_summary):
    """Prints AP, AUROC, and Per-Target Success tables."""
    print("\n" + "="*80)
    print(f"{'DETAILED METRICS BY TARGET':^80}")
    print("="*80)
    
    # Pivot for Hits@10 display
    pivot_hits = plot_df.pivot(index='target_id', columns='Method', values='Hits@10')
    print("\n[Binders Found in Top 10 Predictions]")
    print(pivot_hits.to_string())
    
    print("\n" + "="*80)
    print(f"{'GLOBAL SUCCESS RATE SUMMARY (Targets with >=1 Binder)':^80}")
    print("="*80)
    print(success_summary.to_string(index=False))
    print("="*80 + "\n")

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
    success_stats = []

    for label, (col, lower_is_better) in METRICS_CONFIG.items():
        if col not in merged_df.columns: continue
        
        target_hits_1_count = 0
        target_hits_10_count = 0
        
        for t in targets:
            group = merged_df[merged_df['target_id'] == t].copy()
            scores = -group[col] if lower_is_better else group[col]
            
            # Ranking metrics
            ap_score = get_ap(group['is_binder'], scores)
            auroc_score = get_auroc(group['is_binder'], scores)
            
            # Per-target binder counts
            group_sorted = group.sort_values(by=col, ascending=lower_is_better)
            h1 = int(group_sorted.head(1)['is_binder'].sum())
            h10 = int(group_sorted.head(10)['is_binder'].sum())
            
            plot_data_list.append({
                'target_id': t, 
                'AP': ap_score, 
                'AUROC': auroc_score, 
                'Hits@1': h1, 
                'Hits@10': h10, 
                'Method': label
            })
            
            if h1 >= 1: target_hits_1_count += 1
            if h10 >= 1: target_hits_10_count += 1
            
        success_stats.append({
            'Method': label,
            'Global Success@1': target_hits_1_count / len(targets),
            'Global Success@10': target_hits_10_count / len(targets)
        })

    plot_df = pd.DataFrame(plot_data_list)
    success_df = pd.DataFrame(success_stats)

    # 1. Terminal Output
    print_terminal_metrics(plot_df, success_df)
    
    # 2. Per-Target Plots (AP, AUROC, Hits@1, Hits@10)
    for m in ['AP', 'AUROC', 'Hits@1', 'Hits@10']:
        generate_per_target_plot(plot_df, output_dir, metric=m)
    
    # 3. Global Success Rate Plot
    plot_global_success(success_df, output_dir)
    
    # 4. Global Curves
    plot_global_pr_curve(merged_df, METRICS_CONFIG, output_dir)

def generate_per_target_plot(df, output_dir, metric):
    sns.set_context("paper", font_scale=PLOT_SETTINGS['font_scale'])
    plt.figure(figsize=(PLOT_SETTINGS['figsize_width'], max(6, len(df['target_id'].unique()) * PLOT_SETTINGS['bar_height_per_target'])))
    ax = sns.barplot(data=df, x=metric, y='target_id', hue='Method', edgecolor='black', palette='muted')
    
    # Dynamic formatting: floats for AP/AUROC, ints for Hits
    fmt = '%.2f' if metric in ['AP', 'AUROC'] else '%d'
    for container in ax.containers: ax.bar_label(container, fmt=fmt, padding=3)
    
    plt.title(f'Per-Target {metric}', weight='bold')
    if metric in ['AP', 'AUROC']: plt.xlim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"per_target_{metric.lower().replace('@','_')}.png"), dpi=300)
    plt.close()

def plot_global_success(success_df, output_dir):
    melted = success_df.melt(id_vars='Method', var_name='Metric', value_name='Rate')
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=melted, x='Metric', y='Rate', hue='Method', palette='viridis')
    for container in ax.containers: ax.bar_label(container, fmt='%.2f', padding=3)
    plt.title('Global Success Rate (Fraction of targets with >=1 binder found)', weight='bold')
    plt.ylim(0, 1.1); plt.ylabel('Success Rate'); plt.savefig(os.path.join(output_dir, "global_success_rates.png"), dpi=300)
    plt.close()

def plot_global_pr_curve(merged_df, metrics_config, output_dir):
    plt.figure(figsize=(10, 8))
    for label, (col, lower_is_better) in metrics_config.items():
        if col not in merged_df.columns: continue
        y_true, y_scores = pd.to_numeric(merged_df['is_binder'], errors='coerce'), pd.to_numeric(merged_df[col], errors='coerce')
        if lower_is_better: y_scores = -y_scores 
        mask = y_true.notna() & y_scores.notna()
        if y_true[mask].nunique() < 2: continue
        precision, recall, _ = precision_recall_curve(y_true[mask], y_scores[mask])
        plt.plot(recall, precision, lw=2.5, label=f'{label} (AP = {get_ap(y_true, y_scores):.2f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "global_pr_curve.png"), dpi=300); plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    target_path = args.path if not os.path.isdir(args.path) else os.path.join(args.path, "predictions.csv")
    analyze_results(target_path, os.path.dirname(target_path))