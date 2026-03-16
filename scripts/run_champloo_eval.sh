#!/bin/bash
#SBATCH --partition=shared
#SBATCH --account=jgray21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00
#SBATCH --output=slogs/%j.out
#SBATCH --job-name=champloo_eval
set -euo pipefail

PROTAFF_DIR=/scratch16/jgray21/lchu11/projects/ProtAff
RESULTS=${PROTAFF_DIR}/results/champloo
OUTPUT=${PROTAFF_DIR}/analysis/paper_v0/champloo

cd "${PROTAFF_DIR}"

# Build args dynamically based on which prediction files exist
ARGS=""

if [ -f "${RESULTS}/protaff_predictions.csv" ]; then
    ARGS="${ARGS} --protaff ${RESULTS}/protaff_predictions.csv"
    echo "Found ProtAff predictions"
else
    echo "WARNING: ProtAff predictions not found, skipping"
fi

if [ -f "${RESULTS}/mint_predictions.csv" ]; then
    ARGS="${ARGS} --mint ${RESULTS}/mint_predictions.csv"
    echo "Found MINT predictions"
else
    echo "WARNING: MINT predictions not found, skipping"
fi

if [ -f "${RESULTS}/crossaffinity_predictions.csv" ]; then
    ARGS="${ARGS} --crossaff ${RESULTS}/crossaffinity_predictions.csv"
    echo "Found CrossAffinity predictions"
else
    echo "WARNING: CrossAffinity predictions not found, skipping"
fi

echo "=== Champloo evaluation ==="
python analysis/eval_champloo.py ${ARGS} -o "${OUTPUT}"

echo "Done. Figures at: ${OUTPUT}/"
