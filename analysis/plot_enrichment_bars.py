import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse

# ================= CONFIGURATION (DEFAULTS) =================
DEFAULT_STRUCT_SCORES_CSV = "data/boltz2/adaptyv/all_models_scores.csv" 
DEFAULT_GT_CSV = "data/test/test_adaptyv.csv"                            

# Aggregation for structural scores
SELECTED_AGG = 'min'  

# Define directionality: True if Higher Score = Better Binder
METRIC_DIRECTIONS = {
    'ipSAE': True,
    'ipTM_af': True,
    'pDockQ': True,
    'pDockQ2': True,
    'LIS': True,
    'predicted_affinity': False # Usually Lower Kd/Energy is better
}
# =================================================

def load_and_prep_data(pred_csv, gt_csv, struct_csv, binder_threshold):
    try:
        struct_df = pd.read_csv(struct_csv)
        gt_df = pd.read_csv(gt_csv)
        pred_df = pd.read_csv(pred_csv)
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
    merged['is_binder'] = (merged['log_Aff'] <= binder_threshold).astype(int)
    
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

def plot_grouped_bar(df_plot, base_rate, output_dir):
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
    
    out_file = os.path.join(output_dir, "validation_EnrichmentBarChart.png")
    plt.savefig(out_file, dpi=300)
    print(f"✅ Saved plot to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Enrichment Bar Plots.")
    parser.add_argument("path", type=str, help="Path to predictions csv OR directory containing predictions.csv")
    parser.add_argument("--gt", type=str, default=DEFAULT_GT_CSV, help="Path to Ground Truth CSV")
    parser.add_argument("--struct", type=str, default=DEFAULT_STRUCT_SCORES_CSV, help="Path to Structural Scores CSV")
    parser.add_argument("--threshold", type=float, default=3.0, help="Binder Threshold (log_Aff <= X is binder)")
    
    args = parser.parse_args()
    
    target_path = args.path
    
    # Auto-infer logic
    if os.path.isdir(target_path):
        potential_file = os.path.join(target_path, "predictions.csv")
        if os.path.exists(potential_file):
            target_path = potential_file
        else:
            print(f"❌ Error: Directory provided but 'predictions.csv' not found in: {target_path}")
            sys.exit(1)
            
    # Derive output directory from the final file path
    output_dir = os.path.dirname(target_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"[Analysis] Processing: {target_path}")
    print(f"[Analysis] Output Dir: {output_dir}")

    df = load_and_prep_data(target_path, args.gt, args.struct, args.threshold)
    
    if df is not None:
        stats_df, random_rate = calculate_enrichment_stats(df)
        plot_grouped_bar(stats_df, random_rate, output_dir)