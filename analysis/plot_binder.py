import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
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
    'figsize_width': 14,
    'bar_height_per_target': 0.7,
}

# Highlight color for our method
HIGHLIGHT_COLOR = '#D55E00'
BASELINE_COLOR = '#0072B2'
# =================================================

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

def get_ap(y_true, y_scores):
    mask = y_true.notna() & y_scores.notna()
    y_t, y_s = y_true[mask], y_scores[mask]
    if len(y_t) == 0 or y_t.nunique() < 2: return 0.0
    return average_precision_score(y_t, y_s)

def get_auroc(y_true, y_scores):
    mask = y_true.notna() & y_scores.notna()
    y_t, y_s = y_true[mask], y_scores[mask]
    if len(y_t) == 0 or y_t.nunique() < 2: return 0.5 
    return roc_auc_score(y_t, y_s)

def print_metrics_summary(plot_df):
    print("\n" + "="*120)
    print(f"{'DETAILED PER-TARGET METRICS':^120}")
    print("="*120)
    display_df = plot_df.copy().sort_values(['Method', 'target_id'])
    print(display_df.to_string(index=False, formatters={
        'AP': '{:,.3f}'.format, 
        'AUROC': '{:,.3f}'.format,
        'P@10': '{:,.3f}'.format,
        'Success@1': '{:,.0f}'.format,
        'Success@10': '{:,.0f}'.format
    }))

    print("\n" + "="*120)
    print(f"{'AVERAGE METRICS ACROSS ALL TARGETS':^120}")
    print("="*120)
    summary = plot_df.groupby('Method').agg({
        'AP': 'mean', 'AUROC': 'mean', 'P@10': 'mean', 'Success@1': 'mean', 'Success@10': 'mean'
    }).reset_index()
    summary.columns = ['Method', 'Mean AP', 'Mean AUROC', 'Mean P@10', 'Success Rate @1', 'Success Rate @10']
    print(summary.to_string(index=False, formatters={
        'Mean AP': '{:,.3f}'.format,
        'Mean AUROC': '{:,.3f}'.format,
        'Mean P@10': '{:,.3f}'.format,
        'Success Rate @1': '{:,.2%}'.format,
        'Success Rate @10': '{:,.2%}'.format
    }))
    print("="*120 + "\n")

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
        
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
        merged_df['is_binder'] = pd.to_numeric(merged_df['is_binder'], errors='coerce').fillna(0)
        
        for t in targets:
            group = merged_df[merged_df['target_id'] == t].copy()
            scores = -group[col] if lower_is_better else group[col]
            
            ap_score = get_ap(group['is_binder'], scores)
            auroc_score = get_auroc(group['is_binder'], scores)
            
            group_sorted = group.sort_values(by=col, ascending=lower_is_better)
            top_1 = group_sorted.head(1)
            top_10 = group_sorted.head(10)
            
            s1 = 1 if top_1['is_binder'].sum() >= 1 else 0
            s10 = 1 if top_10['is_binder'].sum() >= 1 else 0
            # Precision @ 10: Percentage of binders in the top 10
            p10 = top_10['is_binder'].mean() if len(top_10) > 0 else 0.0
            
            plot_data_list.append({
                'target_id': t, 'Method': label, 'AP': ap_score, 
                'AUROC': auroc_score, 'P@10': p10, 
                'Success@1': s1, 'Success@10': s10
            })

    plot_df = pd.DataFrame(plot_data_list)

    # 1. Console Output
    print_metrics_summary(plot_df)
    
    # 2. Bar Plots (Per-Target)
    for m in ['AP', 'AUROC', 'P@10', 'Success@1', 'Success@10']:
        generate_per_target_plot(plot_df, output_dir, metric=m)
    
    # 3. Bar Plots (Global)
    plot_global_summary(plot_df, output_dir)

    # 4. Distribution Strip Plots
    generate_distribution_plots(merged_df, output_dir)

def generate_per_target_plot(df, output_dir, metric):
    setup_slide_style()
    h = max(6, len(df['target_id'].unique()) * PLOT_SETTINGS['bar_height_per_target'])
    plt.figure(figsize=(PLOT_SETTINGS['figsize_width'], h))

    palette = {m: HIGHLIGHT_COLOR if 'Predicted' in m else BASELINE_COLOR
               for m in df['Method'].unique()}
    ax = sns.barplot(data=df, x=metric, y='target_id', hue='Method',
                     palette=palette, edgecolor='black')

    fmt = '%.2f' if metric in ['AP', 'AUROC', 'P@10'] else '%d'
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=3, fontsize=14)

    plt.title(metric, weight='bold')
    plt.ylabel("")
    plt.legend(frameon=True, framealpha=0.9)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"per_target_{metric.lower().replace('@','_')}.png"), dpi=300)
    plt.close()

def plot_global_summary(plot_df, output_dir):
    setup_slide_style()
    summary_df = plot_df.groupby('Method')[['AP', 'AUROC', 'P@10', 'Success@1', 'Success@10']].mean().reset_index()
    melted = summary_df.melt(id_vars='Method', var_name='metric_name', value_name='Average Value')

    palette = {m: HIGHLIGHT_COLOR if 'Predicted' in m else BASELINE_COLOR
               for m in melted['Method'].unique()}

    plt.figure(figsize=(14, 7))
    ax = sns.barplot(data=melted, x='metric_name', y='Average Value', hue='Method',
                     palette=palette, edgecolor='black')

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=14)

    plt.title('Average Across All Targets', weight='bold')
    plt.ylim(0, 1.1)
    plt.xlabel("")
    plt.ylabel('Score')
    plt.legend(frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_summary_metrics.png"), dpi=300)
    plt.close()

def generate_distribution_plots(merged_df, output_dir):
    """Generates strip plots showing individual design scores per target."""
    setup_slide_style()
    
    plot_df = merged_df.copy()
    
    plot_df['Binding Status'] = plot_df['is_binder'].map({
        True: 'Binder', 
        False: 'Non-Binder',
        1: 'Binder',
        0: 'Non-Binder',
        1.0: 'Binder',
        0.0: 'Non-Binder'
    })
    
    plot_df = plot_df.dropna(subset=['Binding Status', 'target_id'])
    cb_palette = {'Binder': '#009E73', 'Non-Binder': '#D55E00'}

    for label, (col, lower_is_better) in METRICS_CONFIG.items():
        if col not in plot_df.columns:
            continue
            
        temp_df = plot_df.dropna(subset=[col]).copy()
        if temp_df.empty:
            continue

        plot_col = col
        display_label = label
        if lower_is_better:
            temp_df[f'neg_{col}'] = -temp_df[col]
            plot_col = f'neg_{col}'
            display_label = f"-{label} (Higher is Better)"

        h = max(8, len(temp_df['target_id'].unique()) * PLOT_SETTINGS['bar_height_per_target'])
        plt.figure(figsize=(PLOT_SETTINGS['figsize_width'], h))
        
        ax = sns.stripplot(
            data=temp_df, 
            x=plot_col, 
            y='target_id', 
            hue='Binding Status', 
            hue_order=['Non-Binder', 'Binder'],
            palette=cb_palette,
            alpha=0.5, 
            dodge=True, 
            jitter=0.25,
            size=4
        )
        
        plt.title(f'{display_label}', weight='bold', pad=20)
        plt.xlabel(display_label)
        plt.ylabel("")
        plt.grid(axis='x', linestyle='--', alpha=0.3)

        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        
        sns.despine(left=True, bottom=False)
        plt.tight_layout()
        
        safe_name = label.lower().replace(' ', '_').replace('@', '_')
        plt.savefig(os.path.join(output_dir, f"distribution_{safe_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to predictions.csv or directory containing it")
    args = parser.parse_args()
    
    target_path = args.path if not os.path.isdir(args.path) else os.path.join(args.path, "predictions.csv")
    analyze_results(target_path, os.path.dirname(target_path))