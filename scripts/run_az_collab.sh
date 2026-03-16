#!/bin/bash
# Run prediction + analysis for az_collab dataset
#
# Usage: bash scripts/run_az_collab.sh

# === Configuration (edit these) ===
MODEL_PATH="weights/pair_affinity/saved_model_last"  # path to trained model
INPUT_CSV="data/sources/az_collab/predict_input.csv"
OUTPUT_CSV="data/sources/az_collab/predict_output.csv"
GT_COL="fitness"
THRESHOLD="0.0"  # fitness threshold for defining good binders (lower=better)

# === Step 1: Run prediction ===
echo "=== Running inference ==="
python tools/predict.py --model_dir "$MODEL_PATH" \
    --input_csv "$INPUT_CSV" \
    --output_csv "$OUTPUT_CSV"

# === Step 2: Run analysis ===
echo "=== Running analysis ==="
python analysis/plot_scatter.py "$OUTPUT_CSV" \
    --gt-col "$GT_COL" --threshold "$THRESHOLD"
