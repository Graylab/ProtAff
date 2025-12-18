import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= CONFIGURATION =================
STRUCT_SCORES_CSV = "data/boltz2/adaptyv/all_models_scores.csv" # Structural scores (ipSAE, pDockQ, etc.)
GT_CSV = "data/test/test_adaptyv.csv"                           # Ground Truth (id, log_Aff)
#PRED_AFF_CSV = "inference_results/combined_esm2_650M_concat_phase2_from_scratch_2/test_adaptyv/predictions.csv"     # New Predictions (id, predicted_affinity)
PRED_AFF_CSV = "inference_output/cleaned_from_pretrain_1/test_adaptyv/predictions.csv"     # New Predictions (id, predicted_affinity)
OUTPUT_DIR = "analysis_output/enrichment_bar_plots"

# Threshold for "True Binder" (e.g., log_Aff <= -6.0 is <1uM, <= -9.0 is <1nM)
BINDER_THRESHOLD = 3.0 
SELECTED_AGG = 'min'  # Aggregation for structural scores

# Define directionality: True if Higher Score = Better Binder
METRIC_DIRECTIONS = {
    'ipSAE': True,
    'ipTM_af': True,
    'pDockQ': True,
    'pDockQ2': True,
    'LIS': True,
    'predicted_affinity': False # Usually Lower Kd/Energy is better
}

def load_and_prep_data():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    try:
        struct_df = pd.read_csv(STRUCT_SCORES_CSV)
        gt_df = pd.read_csv(GT_CSV)
        pred_df = pd.read_csv(PRED_AFF_CSV)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None

    # Cleanup IDs
    for df in [struct_df, gt_df, pred_df]:
        df['id'] = df['id'].astype(str).str.strip()

    # Aggregate structural scores (e.g., best pose per target)
    score_cols = [c for c in METRIC_DIRECTIONS.keys() if c in struct_df.columns]
    grouped_struct = struct_df.groupby('id')[score_cols].agg(SELECTED_AGG).reset_index()

    # Merge
    merged = pd.merge(grouped_struct, gt_df[['id', 'log_Aff']], on='id', how='inner')
    merged = pd.merge(merged, pred_df[['id', 'predicted_affinity']], on='id', how='inner')

    # Define Ground Truth Binary Label
    merged['is_binder'] = (merged['log_Aff'] <= BINDER_THRESHOLD).astype(int)
    
    return merged

def calculate_enrichment_stats(df):
    plot_data = []
    
    # Base Rate (Random Chance)
    base_rate = df['is_binder'].mean() * 100
    print(f"Base Hit Rate (Random): {base_rate:.2f}%")

    # Define "Top N%" Tiers to simulate different stringency filters
    tiers = [0.05, 0.10, 0.20]
    tier_labels = ["Top 5%", "Top 10%", "Top 20%"]

    # Loop through each method/metric
    for metric, higher_is_better in METRIC_DIRECTIONS.items():
        if metric not in df.columns:
            continue
            
        # Sort dataframe by this metric
        # If Higher is Better: Ascending=False. If Lower is Better: Ascending=True
        sorted_df = df.sort_values(by=metric, ascending=not higher_is_better)
        
        for tier, tier_name in zip(tiers, tier_labels):
            # Select Top N% candidates
            cutoff_idx = int(len(df) * tier)
            if cutoff_idx < 1: cutoff_idx = 1
            
            subset = sorted_df.iloc[:cutoff_idx]
            
            # Calculate % Enriched (Precision)
            enrichment = subset['is_binder'].mean() * 100
            
            plot_data.append({
                "Method": metric,
                "Selection Stringency": tier_name,
                "% Enriched (<1000nM)": enrichment
            })

    return pd.DataFrame(plot_data), base_rate

def plot_grouped_bar(df_plot, base_rate):
    # Setup Scientific Style
    sns.set_theme(style="white", rc={"axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.3})
    plt.figure(figsize=(10, 6))

    # Rename 'predicted_affinity' to 'Ours (Affinity)' for clarity
    df_plot['Method'] = df_plot['Method'].replace({'predicted_affinity': 'Ours (Affinity)'})

    # Custom Palette
    # Highlights 'Ours' vs Baselines
    unique_methods = df_plot['Method'].unique()
    palette = sns.color_palette("Paired", n_colors=len(df_plot['Selection Stringency'].unique()))

    # Plot
    ax = sns.barplot(
        data=df_plot,
        x="Method",
        y="% Enriched (<1000nM)",
        hue="Selection Stringency",
        palette=["#0072B2", "#E69F00", "#D55E00"], # Blue, Orange, Red (Okabe-Ito friendly)
        edgecolor="black",
        linewidth=1.2,
        capsize=0.05
    )

    # Add Random Baseline Line
    plt.axhline(y=base_rate, color='gray', linestyle='--', linewidth=2, label=f'Random ({base_rate:.1f}%)')

    # Formatting
    plt.title("Enrichment of High-Affinity Binders by Method", fontweight='bold', fontsize=14, pad=15)
    plt.ylabel("% Enriched (<1000nM)", fontweight='bold', fontsize=12)
    plt.xlabel("Scoring Method", fontweight='bold', fontsize=12)
    plt.ylim(0, 105) # Keep 0-100% range
    
    # Legend
    plt.legend(title="Filter Stringency", loc='upper left', frameon=True, framealpha=0.9)
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f%%', padding=3, fontsize=10, fontweight='bold')

    plt.tight_layout()
    
    out_file = os.path.join(OUTPUT_DIR, "validation_EnrichmentBarChart.png")
    plt.savefig(out_file, dpi=300)
    print(f"✅ Saved plot to {out_file}")

if __name__ == "__main__":
    df = load_and_prep_data()
    if df is not None:
        stats_df, random_rate = calculate_enrichment_stats(df)
        plot_grouped_bar(stats_df, random_rate)
