#!/bin/bash
# Generate paper figures with ML baseline comparisons.
#
# Usage:
#   bash scripts/add_ml_baselines.sh <predictions.csv>
#
# Optional environment variables:
#   GT_CSV       - ground truth CSV (default: data/test/test_adaptyv.csv)
#   STRUCT_CSV   - AF3 structural scores CSV (default: data/structures/af3/all_models_scores.csv)
#   STRUCT_SEC   - Boltz2 structural scores CSV for supplementary table (default: data/structures/boltz2/adaptyv/all_models_scores.csv)
#   THRESHOLD    - binder classification threshold (default: 3.0)
#   N_BOOT       - bootstrap resamples (default: 1000)
#   OUTPUT_DIR   - output directory (default: analysis/paper)

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/add_ml_baselines.sh <predictions.csv>"
    exit 1
fi

PRED_CSV="$1"
GT_CSV="${GT_CSV:-data/test/test_adaptyv.csv}"
STRUCT_CSV="${STRUCT_CSV:-data/structures/af3/all_models_scores.csv}"
STRUCT_SEC="${STRUCT_SEC:-data/structures/boltz2/adaptyv/all_models_scores.csv}"
THRESHOLD="${THRESHOLD:-3.0}"
N_BOOT="${N_BOOT:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "${PRED_CSV}")/figures}"

MINT_CSV="${MINT_CSV:-results/ml_baselines/mint_test_adaptyv_predictions.csv}"
CROSSAFFINITY_CSV="${CROSSAFFINITY_CSV:-results/ml_baselines/crossaffinity_adaptyv_predictions.csv}"
MARGIN_WEIGHTED_CSV="${MARGIN_WEIGHTED_CSV:-results/model_margin_weighted/predictions.csv}"

ARGS=(
    python analysis/add_ml_baselines.py
    --pred "${PRED_CSV}"
    --baseline "MINT:${MINT_CSV}:interaction_prob:higher"
    --baseline "CrossAffinity:${CROSSAFFINITY_CSV}:pKd:higher"
    --pred-extra "ProtAff (weighted):${MARGIN_WEIGHTED_CSV}"
    --threshold "${THRESHOLD}" --n-boot "${N_BOOT}" -o "${OUTPUT_DIR}"
)

[ -f "${GT_CSV}" ] && ARGS+=(--gt "${GT_CSV}")
[ -f "${STRUCT_CSV}" ] && ARGS+=(--struct "${STRUCT_CSV}")
[ -f "${STRUCT_SEC}" ] && ARGS+=(--struct-secondary "${STRUCT_SEC}")

echo "Running: ${ARGS[*]}"
"${ARGS[@]}"
