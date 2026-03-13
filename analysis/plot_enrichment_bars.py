import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse

# ================= CONFIGURATION (DEFAULTS) =================
DEFAULT_STRUCT_SCORES_CSV = "data/boltz2/adaptyv/all_models_scores.csv"
DEFAULT_STRUCT_AF3_CSV = "data/af3/all_models_scores.csv"
DEFAULT_GT_CSV = "data/test/test_adaptyv.csv"

# Aggregation for structural scores
SELECTED_AGG = 'min'

STRUCT_METRICS = ['ipSAE', 'ipTM_af', 'pDockQ', 'pDockQ2', 'LIS']

# Define directionality: True if Higher Score = Better Binder
# Populated at runtime with source-prefixed keys
BASE_METRIC_DIRECTIONS = {
    'ipSAE': True,
    'ipTM_af': True,
    'pDockQ': True,
    'pDockQ2': True,
    'LIS': True,
}
# =================================================

def load_and_prep_data(pred_csv, gt_csv, struct_csv, binder_threshold, struct_af3_csv=None):
    try:
        gt_df = pd.read_csv(gt_csv)
        pred_df = pd.read_csv(pred_csv)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, {}

    for df in [gt_df, pred_df]:
        df['id'] = df['id'].astype(str).str.strip()

    merged = pd.merge(gt_df[['id', 'log_Aff']], pred_df[['id', 'predicted_affinity']], on='id', how='inner')

    # Build metric directions dynamically from each structural source
    metric_directions = {}

    struct_sources = {'Boltz2': struct_csv}
    if struct_af3_csv:
        struct_sources['AF3'] = struct_af3_csv

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
        merged = pd.merge(merged, grouped, on='id', how='left')
        for m in available:
            metric_directions[f'{source_name}_{m}'] = BASE_METRIC_DIRECTIONS[m]

    # predicted_affinity: lower = better binder
    metric_directions['predicted_affinity'] = False

    # Define Ground Truth Binary Label
    merged['is_binder'] = (merged['log_Aff'] <= binder_threshold).astype(int)

    return merged, metric_directions

def calculate_enrichment_stats(df, metric_directions):
    plot_data = []

    # Base Rate (Random Chance)
    base_rate = df['is_binder'].mean() * 100
    print(f"Base Hit Rate (Random): {base_rate:.2f}%")

    # Define "Top N%" Tiers to simulate different stringency filters
    tiers = [0.05, 0.10, 0.20]
    tier_labels = ["Top 5%", "Top 10%", "Top 20%"]

    # Loop through each method/metric
    for metric, higher_is_better in metric_directions.items():
        if metric not in df.columns:
            continue

        # Sort dataframe by this metric
        sorted_df = df.sort_values(by=metric, ascending=not higher_is_better)

        for tier, tier_name in zip(tiers, tier_labels):
            cutoff_idx = int(len(df) * tier)
            if cutoff_idx < 1: cutoff_idx = 1

            subset = sorted_df.iloc[:cutoff_idx]

            enrichment = subset['is_binder'].mean() * 100

            plot_data.append({
                "Method": metric,
                "Selection Stringency": tier_name,
                "% Enriched (<1000nM)": enrichment
            })

    return pd.DataFrame(plot_data), base_rate

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

def plot_grouped_bar(df_plot, base_rate, output_dir):
    setup_slide_style()
    plt.figure(figsize=(12, 7))

    # Rename 'predicted_affinity' for clarity
    df_plot['Method'] = df_plot['Method'].replace({'predicted_affinity': 'Predicted Affinity'})

    # Plot
    ax = sns.barplot(
        data=df_plot,
        x="Method",
        y="% Enriched (<1000nM)",
        hue="Selection Stringency",
        palette=["#0072B2", "#E69F00", "#D55E00"],
        edgecolor="black",
        linewidth=1.2,
        capsize=0.05
    )

    # Highlight 'Predicted Affinity' x-tick label
    for tick_label in ax.get_xticklabels():
        if 'Predicted Affinity' in tick_label.get_text():
            tick_label.set_fontweight('bold')
            tick_label.set_color('#D55E00')

    # Add Random Baseline Line
    plt.axhline(y=base_rate, color='gray', linestyle=':', linewidth=2, label=f'Random ({base_rate:.1f}%)')

    plt.title("Enrichment of High-Affinity Binders", fontweight='bold', pad=15)
    plt.ylabel("% Enriched (<1000nM)")
    plt.xlabel("")
    plt.ylim(0, 105)

    plt.legend(loc='upper left', frameon=True, framealpha=0.9)

    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f%%', padding=3, fontsize=12, fontweight='bold')

    plt.tight_layout()
    
    out_file = os.path.join(output_dir, "validation_EnrichmentBarChart.png")
    plt.savefig(out_file, dpi=300)
    print(f"✅ Saved plot to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Enrichment Bar Plots.")
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
            print(f"Error: Directory provided but 'predictions.csv' not found in: {target_path}")
            sys.exit(1)

    # Derive output directory from the final file path
    output_dir = os.path.dirname(target_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"[Analysis] Processing: {target_path}")
    print(f"[Analysis] Output Dir: {output_dir}")

    af3_path = args.struct_af3 if args.struct_af3 else None
    df, metric_directions = load_and_prep_data(target_path, args.gt, args.struct, args.threshold, af3_path)

    if df is not None:
        stats_df, random_rate = calculate_enrichment_stats(df, metric_directions)
        plot_grouped_bar(stats_df, random_rate, output_dir)