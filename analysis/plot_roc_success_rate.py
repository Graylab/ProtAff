import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import os

# ================= CONFIGURATION =================
STRUCT_SCORES_CSV = "data/boltz2/adaptyv/all_models_scores.csv" # Structural scores (ipSAE, pDockQ, etc.)
GT_CSV = "data/test/test_adaptyv.csv"                           # Ground Truth (id, log_Aff)
PRED_AFF_CSV = "inference_results/combined_esm2_650M_concat_phase2_from_scratch_2/test_adaptyv/predictions.csv"     # New Predictions (id, predicted_affinity)
OUTPUT_DIR = "analysis_output/roc_success_rate_plots"

# --- CRITICAL: DEFINING 'TRUE BINDERS' ---
# What log_Aff value counts as a binder?
# Example: If log_Aff is log10(Kd in Molar): -6 is 1uM, -9 is 1nM.
# Set this to your specific project threshold. Lower log_Aff = Binder.
BINDER_THRESHOLD = 3.0 

# Aggregation to use for the Structural Scores (Standard is 'max' or 'mean')
# We will use 'max' for this analysis (Best confident model)
SELECTED_AGG = 'min' 

STRUCT_METRICS = ['ipSAE', 'ipTM_af', 'pDockQ', 'pDockQ2', 'LIS']
# =================================================

def analyze_enrichment():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading data...")
    try:
        struct_df = pd.read_csv(STRUCT_SCORES_CSV)
        gt_df = pd.read_csv(GT_CSV)
        aff_pred_df = pd.read_csv(PRED_AFF_CSV)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Cleanup IDs
    struct_df['id'] = struct_df['id'].astype(str).str.strip()
    gt_df['id'] = gt_df['id'].astype(str).str.strip()
    aff_pred_df['id'] = aff_pred_df['id'].astype(str).str.strip()

    # 1. Prepare Data
    grouped_df = struct_df.groupby('id')[STRUCT_METRICS].agg(SELECTED_AGG).reset_index()
    
    # Merge All
    merged_df = pd.merge(grouped_df, gt_df[['id', 'log_Aff']], on='id', how='inner')
    merged_df = pd.merge(merged_df, aff_pred_df[['id', 'predicted_affinity']], on='id', how='inner')

    # 2. Define Binary Class
    merged_df['is_binder'] = (merged_df['log_Aff'] <= BINDER_THRESHOLD).astype(int)
    
    # Transform scores (Higher = Better)
    merged_df['neg_predicted_affinity'] = -1 * merged_df['predicted_affinity']

    num_binders = merged_df['is_binder'].sum()
    total_samples = len(merged_df)
    base_rate = num_binders / total_samples

    print(f"Total Samples: {total_samples}")
    print(f"True Binders (log_Aff <= {BINDER_THRESHOLD}): {num_binders} ({base_rate:.1%} hit rate)")

    # =================================================================
    # --- NEW: Generate Confirmatory Scatter Plot ---
    # This passes the prepared dataframe and your thresholds
    plot_negated_affinity_scatter(merged_df, BINDER_THRESHOLD, OUTPUT_DIR)
    # =================================================================
    
    if num_binders == 0 or num_binders == total_samples:
        print("ERROR: Cannot calculate ROC/Enrichment. 0 binders or 0 non-binders.")
        return

    # Prepare list of metrics
    eval_metrics = []
    for m in STRUCT_METRICS:
        eval_metrics.append((f"{m} ({SELECTED_AGG})", m, True))
    eval_metrics.append(("Predicted Affinity", "neg_predicted_affinity", False))

    # --- DEFINE RESEARCH PAPER COLORS (Okabe-Ito / High Contrast) ---
    research_colors = [
        "#0072B2", # Blue
        "#009E73", # Bluish Green
        "#CC79A7", # Reddish Purple
        "#56B4E9", # Sky Blue
        "#E69F00", # Orange
        "#333333", # Dark Grey
        "#F0E442", # Yellow
    ]
    
    color_map = {}
    color_idx = 0
    
    for label, col, is_struct in eval_metrics:
        if "Predicted Affinity" in label:
            # Highlight: Vermillion (Deep Red-Orange)
            color_map[label] = "#D55E00" 
        else:
            # Cycle through the other research colors
            color_map[label] = research_colors[color_idx % len(research_colors)]
            color_idx += 1

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

        print(f"{label:<25} | {auc:.3f}    | {ef_10:.2f}     | {ef_20:.2f}")

    # Standard Figure Size for both plots
    FIG_SIZE = (10, 7)

    # 4. PLOT 1: ROC Curves
    setup_plotting()
    plt.figure(figsize=FIG_SIZE)
    
    results_sorted = sorted(results, key=lambda x: x['AUC'], reverse=True)
    
    for res in results_sorted:
        label = res['Metric']
        color = color_map[label]
        
        fpr, tpr, _ = metrics.roc_curve(res['y_true'], res['y_score'])
        
        lw = 3 if "Predicted Affinity" in label else 2.5
        zorder = 10 if "Predicted Affinity" in label else 5
        
        plt.plot(fpr, tpr, lw=lw, label=f"{label} (AUC={res['AUC']:.2f})", color=color, zorder=zorder)

    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title(f'ROC Evaluation', fontweight='bold', pad=15)
    
    # Legend Lower Right
    plt.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor='gray')
    
    plt.grid(True, alpha=0.2, linestyle='-')
    
    out_roc = os.path.join(OUTPUT_DIR, "validation_ROC.png")
    plt.savefig(out_roc, dpi=300, bbox_inches='tight')
    print(f"\nSaved ROC Plot: {out_roc}")
    plt.close()

    # 5. PLOT 2: Success Rate (Precision) at Top N%
    plot_success_rates(results, total_samples, base_rate, color_map, FIG_SIZE)
    
    # 6. Save Summary Table
    df_res = pd.DataFrame(results)[['Metric', 'AUC', 'EF_10', 'EF_20']]
    df_res = df_res.sort_values(by='AUC', ascending=False)
    df_res.to_csv(os.path.join(OUTPUT_DIR, "enrichment_summary.csv"), index=False)

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

def plot_success_rates(results, total_n, base_rate, color_map, fig_size):
    """Plots Hit Rate (Precision) vs Top % Screened"""
    plt.figure(figsize=fig_size)
    
    thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]
    threshold_labels = ["Top 5%", "Top 10%", "Top 20%", "Top 30%", "Top 50%"]
    indices = np.arange(len(thresholds))
    
    sorted_results = sorted(results, key=lambda x: x['AUC'], reverse=True)

    for res in sorted_results:
        label = res['Metric']
        color = color_map[label]
        
        precisions = []
        for t in thresholds:
            n = len(res['y_true'])
            cut = int(n * t)
            if cut == 0: cut = 1
            
            sorted_idx = np.argsort(res['y_score'])[::-1]
            top_binders = res['y_true'].iloc[sorted_idx].values[:cut]
            precision = sum(top_binders) / cut
            precisions.append(precision)
            
        plt.plot(indices, precisions, marker='o', linewidth=2.5, markersize=8, 
                 label=label, color=color)

    plt.axhline(y=base_rate, color='#333333', linestyle='--', linewidth=1.5, label=f"Random ({base_rate:.1%})")

    plt.xticks(indices, threshold_labels, fontsize=12)
    plt.ylabel("Precision (Success Rate)", fontweight='bold')
    plt.title("Enrichment Analysis", fontweight='bold', pad=15)
    
    # Legend Lower Right (Inside plot)
    plt.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor='gray')
    
    plt.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "validation_SuccessRate.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved Success Rate Plot: {out_path}")
    plt.close()

def plot_negated_affinity_scatter(df, binder_threshold, output_dir):
    """
    Plots (-Predicted) vs (-log_Aff). 
    Result: Top-Right corner contains the best binders (Higher is Better).
    """
    print("\nGenerating Negated Affinity Scatter Plot (Higher is Better)...")
    
    # 1. Create Negated Columns (Higher = Better)
    # We use .copy() to avoid SettingWithCopy warnings on the slice
    plot_df = df.copy()
    plot_df['neg_log_Aff'] = -1 * plot_df['log_Aff']
    plot_df['neg_pred_Aff'] = -1 * plot_df['predicted_affinity']
    
    # New Threshold: If binder was log_Aff <= -6, now it is neg_log_Aff >= 6
    neg_binder_threshold = -1 * binder_threshold

    # 2. Calculate Top N% Cutoffs (Sort Descending now)
    # Since Higher is Better, we sort highest first
    df_sorted = plot_df.sort_values(by='neg_pred_Aff', ascending=False)
    total_n = len(df_sorted)
    
    thresholds_pct = [0.05, 0.10, 0.20, 0.50]
    cutoff_map = {}
    
    for t in thresholds_pct:
        cutoff_idx = int(total_n * t)
        if cutoff_idx >= total_n: cutoff_idx = total_n - 1
        val = df_sorted.iloc[cutoff_idx]['neg_pred_Aff']
        cutoff_map[f"Top {int(t*100)}%"] = val

    # 3. Plot Setup
    plt.figure(figsize=(12, 9))
    plot_df['Label'] = plot_df['is_binder'].map({1: 'True Binder', 0: 'Non-Binder'})

    # Scatter
    sns.scatterplot(
        data=plot_df, x='neg_pred_Aff', y='neg_log_Aff',
        hue='Label', palette={'True Binder': '#D55E00', 'Non-Binder': '#0072B2'},
        style='Label', markers={'True Binder': 'o', 'Non-Binder': 'X'},
        alpha=0.6, s=80, edgecolor='w', linewidth=0.5, zorder=2
    )

    # 4. Reference Lines
    # A) Horizontal Line: Ground Truth Threshold
    # Note: Binders are now ABOVE this line (>= threshold)
    plt.axhline(y=neg_binder_threshold, color='#CC79A7', linestyle='--', linewidth=2.5, 
                label=f'GT Threshold (pKd ≥ {neg_binder_threshold})', zorder=1)

    # B) Vertical Lines: Predicted Cutoffs
    # Note: Top candidates are to the RIGHT of these lines
    line_colors = ['#009E73', '#E69F00', '#56B4E9', '#333333'] 
    for i, (label, val) in enumerate(cutoff_map.items()):
        plt.axvline(x=val, color=line_colors[i], linestyle=':', linewidth=2, 
                    label=f'{label} Cutoff (Score ≥ {val:.2f})', zorder=1)
        
        # Label at top
        plt.text(val, plt.gca().get_ylim()[1], f'  {label}', 
                 color=line_colors[i], fontsize=10, rotation=90, verticalalignment='top', zorder=3)

    # 5. Highlight the "Golden Corner" (Top-Right)
    # Area: Right of Top 5% cutoff AND Above GT Threshold
    xmin, xmax = plt.gca().get_xlim()
    ymin, ymax = plt.gca().get_ylim()
    
    # Fill region: x=[cutoff, xmax], y=[threshold, ymax]
    top5_val = cutoff_map["Top 5%"]
    
    # We use fill_between to shade the area where y >= threshold
    # constrained by x >= top5_val
    plt.fill_between([top5_val, xmax], neg_binder_threshold, ymax, 
                     color='#009E73', alpha=0.1, zorder=0, label='Ideal Region')

    # Titles
    plt.title("Predicted vs. Experimental Affinity (Negated Scale)", fontweight='bold', pad=20, fontsize=16)
    plt.xlabel("Negative Predicted Affinity (Higher is Better) $\\rightarrow$", fontweight='bold', fontsize=14)
    plt.ylabel("Negative Experimental log_Aff (Higher is Better) $\\rightarrow$", fontweight='bold', fontsize=14)

    # Legend Cleanup
    plt.legend(loc='upper left', frameon=True, framealpha=0.95, shadow=True)
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "validation_NegatedScatter.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved Negated Scatter Plot: {out_path}")
    plt.close()

def setup_plotting():
    """Sets specific visual styles for scientific publication"""
    sns.set_theme(style="whitegrid", rc={
        'axes.edgecolor': '.15',
        'xtick.bottom': True,
        'ytick.left': True,
        'grid.linestyle': '--'
    })
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
        'font.size': 12,              
        'axes.titlesize': 14,         
        'axes.labelsize': 12,         
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'lines.linewidth': 2.5
    })

if __name__ == "__main__":
    analyze_enrichment()# =================================================
