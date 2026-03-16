#!/bin/bash
# Generate paper-quality figures from model predictions.
#
# Usage:
#   bash scripts/make_paper.sh <predictions.csv>
#
# Optional environment variables:
#   GT_CSV       - ground truth CSV (default: data/test/test_adaptyv.csv)
#   STRUCT_CSV   - Boltz2 structural scores CSV (default: data/structures/boltz2/adaptyv/all_models_scores.csv)
#   STRUCT_AF3   - AF3 structural scores CSV (default: data/structures/af3/all_models_scores.csv)
#   THRESHOLD    - binder classification threshold (default: 3.0)
#   OUTPUT_DIR   - output directory (default: analysis/paper)

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/make_paper.sh <predictions.csv>"
    exit 1
fi

PRED_CSV="$1"
GT_CSV="${GT_CSV:-data/test/test_adaptyv.csv}"
STRUCT_CSV="${STRUCT_CSV:-data/structures/boltz2/adaptyv/all_models_scores.csv}"
STRUCT_AF3="${STRUCT_AF3:-data/structures/af3/all_models_scores.csv}"
THRESHOLD="${THRESHOLD:-3.0}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis/paper_v2}"

CMD="python analysis/make_paper.py results --pred ${PRED_CSV} -o ${OUTPUT_DIR} --threshold ${THRESHOLD}"

[ -f "${GT_CSV}" ] && CMD="${CMD} --gt ${GT_CSV}"
[ -f "${STRUCT_CSV}" ] && CMD="${CMD} --struct ${STRUCT_CSV}"
[ -f "${STRUCT_AF3}" ] && CMD="${CMD} --struct-af3 ${STRUCT_AF3}"

echo "Running: ${CMD}"
${CMD}
