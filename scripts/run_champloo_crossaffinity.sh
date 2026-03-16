#!/bin/bash
#SBATCH --partition=a100
#SBATCH --account=jgray21_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=2:00:00
#SBATCH --output=slogs/%j.out
#SBATCH --job-name=champloo_crossaff
set -euo pipefail

# Run CrossAffinity inference on champloo all-vs-all pairs
PROTAFF_DIR=/scratch16/jgray21/lchu11/projects/ProtAff
CROSS_DIR=/scratch16/jgray21/lchu11/projects/CrossAffinity
INPUT_CSV=${PROTAFF_DIR}/data/sources/ab_ag_champloo/champloo_allvsall.csv
CROSS_INPUT=${CROSS_DIR}/champloo_input.csv
CROSS_OUTPUT=${CROSS_DIR}/champloo_predictions.csv
OUTPUT_CSV=${PROTAFF_DIR}/results/champloo/crossaffinity_predictions.csv

if [ ! -f "${INPUT_CSV}" ]; then
    echo "ERROR: ${INPUT_CSV} not found. Run prepare_champloo_pairs.py first."
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT_CSV}")"

echo "=== CrossAffinity inference on champloo benchmark ==="
echo "Input: ${INPUT_CSV}"
echo "Output: ${OUTPUT_CSV}"

# Step 1: Prepare headerless 2-column CSV for CrossAffinity
python -c "
import pandas as pd
df = pd.read_csv('${INPUT_CSV}')
df[['binder_sequence', 'target_sequence']].to_csv('${CROSS_INPUT}', header=False, index=False)
print(f'Prepared {len(df)} samples for CrossAffinity')
"

# Step 2: Run CrossAffinity inference
cd "${CROSS_DIR}"
python inference.py \
    --filepath "${CROSS_INPUT}" \
    --output "${CROSS_OUTPUT}" \
    --batch_size 32 \
    --esm2_device cuda \
    --cross_affinity_device cuda \
    --num_workers 6

# Step 3: Merge predictions back
python -c "
import pandas as pd
gt = pd.read_csv('${INPUT_CSV}')
pred = pd.read_csv('${CROSS_OUTPUT}')
gt['pKd'] = pred['pKd'].values
gt.to_csv('${OUTPUT_CSV}', index=False)
print(f'Saved {len(gt)} merged predictions to ${OUTPUT_CSV}')
"

echo "Done. Results at: ${OUTPUT_CSV}"
