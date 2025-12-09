import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import os

# ================= CONFIGURATION =================
MODEL = "esm2_t33_650M_UR50D"
STRUCT_SCORES_CSV = "data/boltz2/adaptyv/all_models_scores.csv" # Structural scores (ipSAE, pDockQ, etc.)
GT_CSV = "data/test/test_adaptyv.csv"                           # Ground Truth (id, log_Aff)
PRED_AFF_CSV = f"inference_results/{MODEL}/test_adaptyv/predictions.csv"     # New Predictions (id, predicted_affinity)
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
    # Aggregate Structural Scores (using SELECTED_AGG, e.g., 'max')
    grouped_df = struct_df.groupby('id')[STRUCT_METRICS].agg(SELECTED_AGG).reset_index()
    
    # Merge All
    merged_df = pd.merge(grouped_df, gt_df[['id', 'log_Aff']], on='id', how='inner')
    merged_df = pd.merge(merged_df, aff_pred_df[['id', 'predicted_affinity']], on='id', how='inner')

    # 2. Define Binary Class (Binder = 1, Non-Binder = 0)
    # Since lower log_Aff is better, a binder is anything <= Threshold
    merged_df['is_binder'] = (merged_df['log_Aff'] <= BINDER_THRESHOLD).astype(int)
    
    # Transform scores for plotting (Higher score = Better)
    # Predicted affinity needs flipping (-1) so higher is better
    merged_df['neg_predicted_affinity'] = -1 * merged_df['predicted_affinity']

    num_binders = merged_df['is_binder'].sum()
    total_samples = len(merged_df)
    base_rate = num_binders / total_samples

    print(f"Total Samples: {total_samples}")
    print(f"True Binders (log_Aff <= {BINDER_THRESHOLD}): {num_binders} ({base_rate:.1%} hit rate)")
    
    if num_binders == 0 or num_binders == total_samples:
        print("ERROR: Cannot calculate ROC/Enrichment. You either have 0 binders or 0 non-binders.")
        print("Adjust your BINDER_THRESHOLD.")
        return

    # Prepare list of metrics to evaluate (Name, Column, is_structural)
    eval_metrics = []
    for m in STRUCT_METRICS:
        eval_metrics.append((f"{m} ({SELECTED_AGG})", m, True))
    eval_metrics.append(("Predicted Affinity", "neg_predicted_affinity", False))

    # Store results
    results = []

    # 3. Calculate AUC and Enrichment Factors
    print("\n" + "="*60)
    print(f"{'Metric':<25} | {'AUC-ROC':<8} | {'EF 10%':<8} | {'EF 20%':<8}")
    print("="*60)

    for label, col, is_struct in eval_metrics:
        if col not in merged_df.columns: continue
        
        # Drop NaNs
        tmp = merged_df[[col, 'is_binder']].dropna()
        y_true = tmp['is_binder']
        y_score = tmp[col] # Higher is better for all these columns now

        # ROC AUC
        auc = metrics.roc_auc_score(y_true, y_score)

        # Enrichment Factors
        # EF_x% = (Hits in Top x%) / (Total Hits * x%)
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

    # 4. PLOT 1: ROC Curves (The Industry Standard)
    setup_plotting()
    plt.figure(figsize=(12, 10))
    
    for res in results:
        fpr, tpr, _ = metrics.roc_curve(res['y_true'], res['y_score'])
        plt.plot(fpr, tpr, lw=3, label=f"{res['Metric']} (AUC={res['AUC']:.2f})")

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    plt.title(f'ROC Curves (Binder Threshold: log_Aff <= {BINDER_THRESHOLD})', fontweight='bold', pad=20)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    out_roc = os.path.join(OUTPUT_DIR, "validation_ROC.png")
    plt.savefig(out_roc, dpi=300)
    print(f"\nSaved ROC Plot: {out_roc}")
    plt.close()

    # 5. PLOT 2: Success Rate (Precision) at Top N%
    # "If I define my cutoff at the top 10%, what % are binders?"
    plot_success_rates(results, total_samples, base_rate)
    
    # 6. Save Summary Table
    df_res = pd.DataFrame(results)[['Metric', 'AUC', 'EF_10', 'EF_20']]
    df_res = df_res.sort_values(by='AUC', ascending=False)
    df_res.to_csv(os.path.join(OUTPUT_DIR, "enrichment_summary.csv"), index=False)

def calc_enrichment_factor(y_true, y_score, percentile):
    """Calculates Enrichment Factor at a specific top percentile"""
    n = len(y_true)
    cutoff_index = int(n * percentile)
    if cutoff_index == 0: return 0.0
    
    # Sort by score descending
    sorted_indices = np.argsort(y_score)[::-1]
    sorted_y_true = y_true.iloc[sorted_indices].values
    
    hits_in_top = sum(sorted_y_true[:cutoff_index])
    total_hits = sum(y_true)
    
    if total_hits == 0: return 0.0
    
    # EF = (Hits_Top / N_Top) / (Total_Hits / N_Total)
    precision_at_k = hits_in_top / cutoff_index
    global_hit_rate = total_hits / n
    
    return precision_at_k / global_hit_rate

def plot_success_rates(results, total_n, base_rate):
    """Plots Hit Rate (Precision) vs Top % Screened"""
    plt.figure(figsize=(14, 8))
    
    thresholds = [0.05, 0.10, 0.20, 0.30, 0.50] # Top 5%, 10%, etc.
    threshold_labels = ["Top 5%", "Top 10%", "Top 20%", "Top 30%", "Top 50%"]
    
    # Prepare data for bar chart
    bar_width = 0.8 / len(results)
    indices = np.arange(len(thresholds))
    
    # Sort results by AUC so best are first in legend
    sorted_results = sorted(results, key=lambda x: x['AUC'], reverse=True)

    for i, res in enumerate(sorted_results):
        precisions = []
        for t in thresholds:
            # Calculate precision at top t
            n = len(res['y_true'])
            cut = int(n * t)
            if cut == 0: cut = 1
            
            # Sort
            sorted_idx = np.argsort(res['y_score'])[::-1]
            top_binders = res['y_true'].iloc[sorted_idx].values[:cut]
            precision = sum(top_binders) / cut
            precisions.append(precision)
            
        plt.plot(indices, precisions, marker='o', linewidth=3, markersize=10, label=res['Metric'])

    # Plot Random Baseline
    plt.axhline(y=base_rate, color='black', linestyle='--', linewidth=2, label=f"Random Hit Rate ({base_rate:.1%})")

    plt.xticks(indices, threshold_labels, fontsize=14)
    plt.ylabel("Success Rate (Precision)", fontweight='bold')
    plt.title("Binder Success Rate by Selection Threshold", fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "validation_SuccessRate.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved Success Rate Plot: {out_path}")
    plt.close()

def setup_plotting():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 14,              
        'axes.titlesize': 18,         
        'axes.labelsize': 15,         
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'figure.titlesize': 22
    })

if __name__ == "__main__":
    analyze_enrichment()# =================================================
