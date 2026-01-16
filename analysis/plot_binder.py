import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
import os
import sys
import argparse
import numpy as np

# ================= CONFIGURATION =================
GT_CSV = "data/binder_design/test_design.csv" 

METRICS_CONFIG = {
    'AF3 ipSAE': ('af3_ipSAE_min', False),    # Higher is better
    'Predicted Affinity': ('predicted_affinity', True), # Lower is better (negated)
}

PLOT_SETTINGS = {
    'figsize_width': 12,
    'bar_height_per_target': 1.0,
    'font_scale': 1.4
}
# =================================================

def get_ap(y_true, y_scores):
    """Calculates Average Precision after explicitly cleaning NaNs."""
    y_true = pd.to_numeric(y_true, errors='coerce')
    y_scores = pd.to_numeric(y_scores, errors='coerce')
    
    mask = y_true.notna() & y_scores.notna()
    y_true_clean = y_true[mask]
    y_scores_clean = y_scores[mask]
    
    if len(y_true_clean) == 0 or y_true_clean.nunique() < 2:
        return 0.0
        
    return average_precision_score(y_true_clean, y_scores_clean)

def get_auroc(y_true, y_scores):
    """Calculates AUROC after explicitly cleaning NaNs."""
    y_true = pd.to_numeric(y_true, errors='coerce')
    y_scores = pd.to_numeric(y_scores, errors='coerce')
    
    mask = y_true.notna() & y_scores.notna()
    y_true_clean = y_true[mask]
    y_scores_clean = y_scores[mask]
    
    if len(y_true_clean) == 0 or y_true_clean.nunique() < 2:
        return 0.0
        
    return roc_auc_score(y_true_clean, y_scores_clean)

def print_terminal_metrics(final_df, merged_df):
    """Prints AP, AUROC and Success Rate tables to the terminal."""
    print("\n" + "="*70)
    print(f"{'METRIC SUMMARY: AVERAGE PRECISION (AP)':^70}")
    print("="*70)
    
    # Pivot using 'target_id' as index
    pivot_ap = final_df.pivot(index='target_id', columns='Method', values='AP')
    print(pivot_ap.to_string(float_format=lambda x: f"{x:.4f}"))
    
    print("\n" + "="*70)
    print(f"{'METRIC SUMMARY: AUROC':^70}")
    print("="*70)
    
    pivot_auroc = final_df.pivot(index='target_id', columns='Method', values='AUROC')
    print(pivot_auroc.to_string(float_format=lambda x: f"{x:.4f}"))
    
    print("\n" + "="*70)
    print(f"{'BINDER SUCCESS RATE (Found in Top 10 Predictions)':^70}")
    print("="*70)
    
    success_rows = []
    # Identify unique real targets
    targets = [t for t in merged_df['target_id'].unique() if pd.notna(t)]
    
    for t in targets:
        row = {'Target': t}
        group = merged_df[merged_df['target_id'] == t]
        for label, (col, lower_is_better) in METRICS_CONFIG.items():
            if col not in group.columns: continue
            
            valid_group = group.dropna(subset=[col, 'is_binder'])
            if valid_group.empty:
                row[label] = "0/0"
                continue
                
            top_10 = valid_group.sort_values(by=col, ascending=lower_is_better).head(10)
            hits = int(top_10['is_binder'].sum())
            row[label] = f"{hits}/{len(top_10)}"
        success_rows.append(row)
    
    print(pd.DataFrame(success_rows).to_string(index=False))
    print("="*70 + "\n")

def analyze_results(pred_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    gt_df = pd.read_csv(GT_CSV)
    pred_df = pd.read_csv(pred_csv)

    # Standardize 'id' for merging, but use 'target_id' for grouping
    gt_df['id'] = gt_df['id'].astype(str).str.strip()
    pred_df['id'] = pred_df['id'].astype(str).str.strip()

    cols_to_merge = ['id'] + [v[0] for v in METRICS_CONFIG.values() if v[0] in pred_df.columns]
    merged_df = gt_df.merge(pred_df[cols_to_merge], on='id', how='left')

    # --- EXCLUDE ROWS WHERE target_id IS NaN ---
    initial_count = len(merged_df)
    merged_df = merged_df.dropna(subset=['target_id'])
    
    if len(merged_df) < initial_count:
        print(f"[Warning] Dropped {initial_count - len(merged_df)} rows with missing target_id.")

    # 1. Visualization
    plot_global_pr_curve(merged_df, METRICS_CONFIG, output_dir)
    plot_global_roc_curve(merged_df, METRICS_CONFIG, output_dir)

    # 2. Stats Calculation
    plot_data_list = []
    targets = merged_df['target_id'].unique()
    
    for t in targets:
        group = merged_df[merged_df['target_id'] == t]
        for label, (col, lower_is_better) in METRICS_CONFIG.items():
            if col not in group.columns: continue
            scores = -group[col] if lower_is_better else group[col]
            ap_score = get_ap(group['is_binder'], scores)
            auroc_score = get_auroc(group['is_binder'], scores)
            plot_data_list.append({
                'target_id': t, 
                'AP': ap_score, 
                'AUROC': auroc_score,
                'Method': label
            })

    plot_df = pd.DataFrame(plot_data_list)
    
    aggs = []
    if not plot_df.empty:
        mean_scores = plot_df.groupby('Method')[['AP', 'AUROC']].mean()
        for label in mean_scores.index:
            aggs.append({
                'target_id': 'Average (Mean)', 
                'AP': mean_scores.loc[label, 'AP'],
                'AUROC': mean_scores.loc[label, 'AUROC'],
                'Method': label
            })

    for label, (col, lower_is_better) in METRICS_CONFIG.items():
        if col not in merged_df.columns: continue
        scores_all = -merged_df[col] if lower_is_better else merged_df[col]
        aggs.append({
            'target_id': 'Global (Pooled)', 
            'AP': get_ap(merged_df['is_binder'], scores_all),
            'AUROC': get_auroc(merged_df['is_binder'], scores_all),
            'Method': label
        })

    final_df = pd.concat([plot_df, pd.DataFrame(aggs)], ignore_index=True)

    # 3. Final Outputs
    print_terminal_metrics(final_df, merged_df)
    save_summary(final_df, output_dir)
    generate_plot(final_df, output_dir, metric='AP')
    generate_plot(final_df, output_dir, metric='AUROC')

def plot_global_pr_curve(merged_df, metrics_config, output_dir):
    plt.figure(figsize=(10, 8))
    
    for label, (col, lower_is_better) in metrics_config.items():
        if col not in merged_df.columns: continue
        y_true = pd.to_numeric(merged_df['is_binder'], errors='coerce')
        y_scores = pd.to_numeric(merged_df[col], errors='coerce')
        if lower_is_better: y_scores = -y_scores 
        mask = y_true.notna() & y_scores.notna()
        if y_true[mask].nunique() < 2: continue
        precision, recall, _ = precision_recall_curve(y_true[mask], y_scores[mask])
        plt.plot(recall, precision, lw=2.5, label=f'{label} (AP = {get_ap(y_true, y_scores):.2f})')
    plt.xlabel('Recall', weight='bold'); plt.ylabel('Precision', weight='bold')
    plt.title('Global PR Curve', weight='bold'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "global_pr_curve.png"), dpi=300); plt.close()

def plot_global_roc_curve(merged_df, metrics_config, output_dir):
    """Plot global ROC curve for all methods."""
    plt.figure(figsize=(10, 8))
    
    for label, (col, lower_is_better) in metrics_config.items():
        if col not in merged_df.columns: continue
        y_true = pd.to_numeric(merged_df['is_binder'], errors='coerce')
        y_scores = pd.to_numeric(merged_df[col], errors='coerce')
        if lower_is_better: y_scores = -y_scores 
        mask = y_true.notna() & y_scores.notna()
        if y_true[mask].nunique() < 2: continue
        fpr, tpr, _ = roc_curve(y_true[mask], y_scores[mask])
        plt.plot(fpr, tpr, lw=2.5, label=f'{label} (AUROC = {get_auroc(y_true, y_scores):.2f})')
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUROC = 0.50)')
    plt.xlabel('False Positive Rate', weight='bold')
    plt.ylabel('True Positive Rate', weight='bold')
    plt.title('Global ROC Curve', weight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "global_roc_curve.png"), dpi=300)
    plt.close()

def save_summary(df, output_dir):
    df.to_csv(os.path.join(output_dir, "metrics_summary_full.csv"), index=False)

def generate_plot(final_df, output_dir, metric='AP'):
    """Generate bar plot for specified metric (AP or AUROC)."""
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=PLOT_SETTINGS['font_scale'])
    num_targets = len(final_df['target_id'].unique())
    plt.figure(figsize=(PLOT_SETTINGS['figsize_width'], max(6, num_targets * PLOT_SETTINGS['bar_height_per_target'])))
    ax = sns.barplot(data=final_df, x=metric, y='target_id', hue='Method', edgecolor='black', palette='muted')
    for container in ax.containers: ax.bar_label(container, fmt='%.2f', padding=3)
    plt.title(f'{metric} Comparison', weight='bold')
    plt.xlim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{metric.lower()}_comparison_final.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    target_path = args.path
    if os.path.isdir(target_path):
        target_path = os.path.join(target_path, "predictions.csv")
    analyze_results(target_path, os.path.dirname(target_path))