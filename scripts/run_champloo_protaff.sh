#!/bin/bash
#SBATCH --partition=a100
#SBATCH --account=jgray21_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=1:00:00
#SBATCH --output=slogs/%j.out
#SBATCH --job-name=champloo_protaff
set -euo pipefail

# Run ProtAff inference on champloo all-vs-all pairs
PROTAFF_DIR=/scratch16/jgray21/lchu11/projects/ProtAff
MODEL_DIR=${PROTAFF_DIR}/weights/pair_affinity/model_v0
INPUT_CSV=${PROTAFF_DIR}/data/sources/ab_ag_champloo/champloo_allvsall.csv
OUTPUT_CSV=${PROTAFF_DIR}/results/champloo/protaff_predictions.csv

cd "${PROTAFF_DIR}"

# Prepare all-vs-all CSV if not already done
if [ ! -f "${INPUT_CSV}" ]; then
    echo "Preparing all-vs-all pairs..."
    python tools/prepare_champloo_pairs.py --output "${INPUT_CSV}"
fi

echo "=== ProtAff inference on champloo benchmark ==="
echo "Input: ${INPUT_CSV}"
echo "Model: ${MODEL_DIR}"
echo "Output: ${OUTPUT_CSV}"

python tools/predict.py \
    --model_dir "${MODEL_DIR}" \
    --input_csv "${INPUT_CSV}" \
    --output_csv "${OUTPUT_CSV}" \
    --batch_size 32

echo "Done. Results at: ${OUTPUT_CSV}"
