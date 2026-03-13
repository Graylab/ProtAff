import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import os
import sys
import argparse

# ================= CONFIGURATION (DEFAULTS) =================
DEFAULT_STRUCT_SCORES_CSV = "data/boltz2/adaptyv/all_models_scores.csv"
DEFAULT_STRUCT_AF3_CSV = "data/af3/all_models_scores.csv"
DEFAULT_GT_CSV = "data/test/test_adaptyv.csv"

# Aggregation to use for the Structural Scores
SELECTED_AGG = 'min'

STRUCT_METRICS = ['ipSAE', 'ipTM_af', 'pDockQ', 'pDockQ2', 'LIS']
# =================================================

def analyze_enrichment(pred_csv, output_dir, gt_csv, struct_csv, binder_threshold, struct_af3_csv=None):
    print(f"\n[Analysis] Predictions: {pred_csv}")
    print(f"[Analysis] Output Dir : {output_dir}")
    print(f"[Analysis] GT File    : {gt_csv}")
    print(f"[Analysis] Threshold  : log_Aff <= {binder_threshold}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading data...")
    try:
        gt_df = pd.read_csv(gt_csv)
        aff_pred_df = pd.read_csv(pred_csv)
    except FileNotFoundError as e:
        print(f"[Error] File not found: {e}")
        return

    gt_df['id'] = gt_df['id'].astype(str).str.strip()
    aff_pred_df['id'] = aff_pred_df['id'].astype(str).str.strip()

    # 1. Prepare Data — load and aggregate each structural source
    struct_sources = {'Boltz2': struct_csv}
    if struct_af3_csv:
        struct_sources['AF3'] = struct_af3_csv

    merged_df = pd.merge(gt_df[['id', 'log_Aff']], aff_pred_df[['id', 'predicted_affinity']], on='id', how='inner')
    eval_metrics = []

    for source_name, csv_path in struct_sources.items():
        try:
            src_df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"[Warning] {source_name} structural scores not found: {csv_path}")
            continue
        src_df['id'] = src_df['id'].astype(str).str.strip()
        available = [m for m in STRUCT_METRICS if m in src_df.columns]
        grouped = src_df.groupby('id')[available].agg(SELECTED_AGG).reset_index()
        grouped.columns = ['id'] + [f'{source_name}_{m}' for m in available]
        merged_df = pd.merge(merged_df, grouped, on='id', how='left')
        for m in available:
            eval_metrics.append((f"{source_name} {m} ({SELECTED_AGG})", f'{source_name}_{m}', True))

    # 2. Define Binary Class
    merged_df['is_binder'] = (merged_df['log_Aff'] <= binder_threshold).astype(int)

    # predicted_affinity is lower=better (log_Kd-like); negate so higher=better for sorting/AUC
    merged_df['predicted_affinity'] = -merged_df['predicted_affinity']

    num_binders = merged_df['is_binder'].sum()
    total_samples = len(merged_df)
    base_rate = num_binders / total_samples

    print(f"Total Samples: {total_samples}")
    print(f"True Binders (log_Aff <= {binder_threshold}): {num_binders} ({base_rate:.1%} hit rate)")

    # =================================================================
    # --- Generate Confirmatory Scatter Plot ---
    plot_negated_affinity_scatter(merged_df, binder_threshold, output_dir)
    # =================================================================

    if num_binders == 0 or num_binders == total_samples:
        print("ERROR: Cannot calculate ROC/Enrichment. 0 binders or 0 non-binders.")
        return

    eval_metrics.append(("Predicted Affinity", "predicted_affinity", False))

    # --- DEFINE RESEARCH PAPER COLORS ---
    boltz2_colors = ["#0072B2", "#CC79A7", "#56B4E9", "#E69F00", "#333333"]
    af3_colors = ["#009E73", "#2CA02C", "#17BECF", "#BCBD22", "#7F7F7F"]

    color_map = {}
    boltz2_idx = 0
    af3_idx = 0

    for label, col, is_struct in eval_metrics:
        if "Predicted Affinity" in label:
            color_map[label] = "#D55E00"
        elif label.startswith("AF3"):
            color_map[label] = af3_colors[af3_idx % len(af3_colors)]
            af3_idx += 1
        else:
            color_map[label] = boltz2_colors[boltz2_idx % len(boltz2_colors)]
            boltz2_idx += 1

    # Store results
    results = []

    # 3. Calculate AUC and Enrichment Factors
    print("\n" + "="*60)
    print(f"{'Metric':<25} | {'AUC-ROC':<8} | {'EF 10%':<8} | {'EF 20%':<8}")
    print("="*60)

    for label, col, is_struct in eval_metrics:
        if col not in merged_df.columns: continue
        
        tmp = merged_df[[col, 'is_binder']].dropna()
        y_true = tmp['is_binder']
        y_score = tmp[col]

        auc = metrics.roc_auc_score(y_true, y_score)
        ef_10 = calc_enrichment_factor(y_true, y_score, 0.10)
        ef_20 = calc_enrichment_factor(y_true, y_score, 0.20)

        results.append({
            'Metric': label,
            'AUC': auc,
            'EF_10': ef_10,
            'EF_20': ef_20,
            'y_true': y_true,
            'y_score': y_score
        })

        print(f"{label:<25} | {auc:.3f}    | {ef_10:.2f}      | {ef_20:.2f}")

    # Standard Figure Size
    FIG_SIZE = (10, 7)

    # 4. PLOT 1: ROC Curves
    setup_slide_style()
    plt.figure(figsize=FIG_SIZE)
    
    results_sorted = sorted(results, key=lambda x: x['AUC'], reverse=True)
    
    for res in results_sorted:
        label = res['Metric']
        color = color_map[label]
        
        fpr, tpr, _ = metrics.roc_curve(res['y_true'], res['y_score'])
        
        is_ours = "Predicted Affinity" in label
        lw = 3.5 if is_ours else 2
        zorder = 10 if is_ours else 5
        ls = '-' if is_ours else '--'

        plt.plot(fpr, tpr, lw=lw, linestyle=ls, label=f"{label} (AUC={res['AUC']:.2f})", color=color, zorder=zorder)

    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves', fontweight='bold', pad=15)

    plt.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor='gray')
    plt.grid(True, alpha=0.2, linestyle='-')
    
    out_roc = os.path.join(output_dir, "validation_ROC.png")
    plt.savefig(out_roc, dpi=300, bbox_inches='tight')
    print(f"\nSaved ROC Plot: {out_roc}")
    plt.close()

    # 5. PLOT 2: Success Rate (Precision) at Top N%
    plot_success_rates(results, total_samples, base_rate, color_map, FIG_SIZE, output_dir)
    
    # 6. Save Summary Table
    df_res = pd.DataFrame(results)[['Metric', 'AUC', 'EF_10', 'EF_20']]
    df_res = df_res.sort_values(by='AUC', ascending=False)
    df_res.to_csv(os.path.join(output_dir, "enrichment_summary.csv"), index=False)

def calc_enrichment_factor(y_true, y_score, percentile):
    """Calculates Enrichment Factor at a specific top percentile"""
    n = len(y_true)
    cutoff_index = int(n * percentile)
    if cutoff_index == 0: return 0.0
    
    sorted_indices = np.argsort(y_score)[::-1]
    sorted_y_true = y_true.iloc[sorted_indices].values
    
    hits_in_top = sum(sorted_y_true[:cutoff_index])
    total_hits = sum(y_true)
    
    if total_hits == 0: return 0.0
    
    precision_at_k = hits_in_top / cutoff_index
    global_hit_rate = total_hits / n
    return precision_at_k / global_hit_rate

def plot_success_rates(results, total_n, base_rate, color_map, fig_size, output_dir):
    """Plots Hit Rate (Precision) vs Top % Screened"""
    plt.figure(figsize=fig_size)
    
    thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]
    threshold_labels = ["Top 5%", "Top 10%", "Top 20%", "Top 30%", "Top 50%"]
    indices = np.arange(len(thresholds))
    
    sorted_results = sorted(results, key=lambda x: x['AUC'], reverse=True)

    for res in sorted_results:
        label = res['Metric']
        color = color_map[label]
        is_ours = "Predicted Affinity" in label

        precisions = []
        for t in thresholds:
            n = len(res['y_true'])
            cut = int(n * t)
            if cut == 0: cut = 1

            sorted_idx = np.argsort(res['y_score'])[::-1]
            top_binders = res['y_true'].iloc[sorted_idx].values[:cut]
            precision = sum(top_binders) / cut
            precisions.append(precision)

        lw = 3.5 if is_ours else 2
        ms = 10 if is_ours else 7
        ls = '-' if is_ours else '--'
        plt.plot(indices, precisions, marker='o', linewidth=lw, markersize=ms,
                 linestyle=ls, label=label, color=color,
                 zorder=10 if is_ours else 5)

    plt.axhline(y=base_rate, color='#333333', linestyle=':', linewidth=1.5, label=f"Random ({base_rate:.1%})")

    plt.xticks(indices, threshold_labels)
    plt.ylabel("Success Rate")
    plt.title("Success Rate by Selection Stringency", fontweight='bold', pad=15)

    plt.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor='gray')
    
    plt.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "validation_SuccessRate.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved Success Rate Plot: {out_path}")
    plt.close()

def plot_negated_affinity_scatter(df, binder_threshold, output_dir):
    """
    Plots predicted_affinity vs (-log_Aff).
    Result: Top-Right corner contains the best binders (Higher is Better for both axes).
    """
    print("\nGenerating Affinity Scatter Plot (Higher is Better)...")
    setup_slide_style()

    # 1. Create columns (Higher = Better)
    plot_df = df.copy()
    plot_df['neg_log_Aff'] = -1 * plot_df['log_Aff']

    neg_binder_threshold = -1 * binder_threshold

    # 2. Calculate Top N% Cutoffs (Sort Descending by predicted_affinity)
    df_sorted = plot_df.sort_values(by='predicted_affinity', ascending=False)
    total_n = len(df_sorted)

    thresholds_pct = [0.05, 0.10, 0.20, 0.50]
    cutoff_map = {}

    for t in thresholds_pct:
        cutoff_idx = int(total_n * t)
        if cutoff_idx >= total_n: cutoff_idx = total_n - 1
        val = df_sorted.iloc[cutoff_idx]['predicted_affinity']
        cutoff_map[f"Top {int(t*100)}%"] = val

    # 3. Plot Setup
    plt.figure(figsize=(12, 9))
    plot_df['Label'] = plot_df['is_binder'].map({1: 'True Binder', 0: 'Non-Binder'})

    # Scatter
    sns.scatterplot(
        data=plot_df, x='predicted_affinity', y='neg_log_Aff',
        hue='Label', palette={'True Binder': '#D55E00', 'Non-Binder': '#0072B2'},
        style='Label', markers={'True Binder': 'o', 'Non-Binder': 'X'},
        alpha=0.6, s=80, edgecolor='w', linewidth=0.5, zorder=2
    )

    # 4. Reference Lines
    plt.axhline(y=neg_binder_threshold, color='#CC79A7', linestyle='--', linewidth=2.5,
                label=f'GT Threshold (pKd >= {neg_binder_threshold})', zorder=1)

    line_colors = ['#009E73', '#E69F00', '#56B4E9', '#333333']
    for i, (label, val) in enumerate(cutoff_map.items()):
        plt.axvline(x=val, color=line_colors[i], linestyle=':', linewidth=2,
                    label=f'{label} Cutoff (Score >= {val:.2f})', zorder=1)

        plt.text(val, plt.gca().get_ylim()[1], f'  {label}',
                 color=line_colors[i], fontsize=14, rotation=90, verticalalignment='top', zorder=3)

    # 5. Highlight "Golden Corner"
    xmin, xmax = plt.gca().get_xlim()
    ymin, ymax = plt.gca().get_ylim()
    top5_val = cutoff_map["Top 5%"]

    plt.fill_between([top5_val, xmax], neg_binder_threshold, ymax,
                     color='#009E73', alpha=0.1, zorder=0, label='Ideal Region')

    plt.title("Predicted vs. Experimental Affinity", fontweight='bold', pad=20)
    plt.xlabel("-Predicted Affinity (Higher is Better)")
    plt.ylabel("-Experimental log_Aff (Higher is Better)")

    plt.legend(loc='upper left', frameon=True, framealpha=0.95, shadow=True)
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "validation_NegatedScatter.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved Negated Scatter Plot: {out_path}")
    plt.close()

def setup_slide_style():
    """Sets visual styles sized for presentation slides."""
    sns.set_theme(style="whitegrid", rc={
        'axes.edgecolor': '.15',
        'xtick.bottom': True,
        'ytick.left': True,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Enrichment (ROC, Success Rate).")
    parser.add_argument("path", type=str, help="Path to predictions csv OR directory containing predictions.csv")
    parser.add_argument("--gt", type=str, default=DEFAULT_GT_CSV, help="Path to Ground Truth CSV")
    parser.add_argument("--struct", type=str, default=DEFAULT_STRUCT_SCORES_CSV, help="Path to Structural Scores CSV")
    parser.add_argument("--struct-af3", type=str, default=DEFAULT_STRUCT_AF3_CSV,
                        help="Path to AF3 structural scores CSV (set to '' to disable)")
    parser.add_argument("--threshold", type=float, default=3.0, help="Binder Threshold (log_Aff <= X is binder)")

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

    af3_path = args.struct_af3 if args.struct_af3 else None
    analyze_enrichment(target_path, output_dir, args.gt, args.struct, args.threshold, af3_path)