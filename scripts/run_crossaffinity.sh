#!/bin/bash
set -euo pipefail


CROSS_DIR=/scratch16/jgray21/lchu11/projects/CrossAffinity
PROTAFF_DIR=/scratch16/jgray21/lchu11/projects/ProtAff
INPUT_CSV=${CROSS_DIR}/adaptyv_input.csv
OUTPUT_CSV=${CROSS_DIR}/adaptyv_predictions.csv

# --- Step 1: Prepare input (headerless 2-column CSV) ---
python -c "
import pandas as pd
df = pd.read_csv('${PROTAFF_DIR}/data/test/test_adaptyv.csv')
df[['binder_sequence', 'target_sequence']].to_csv('${INPUT_CSV}', header=False, index=False)
print(f'Prepared {len(df)} samples for CrossAffinity')
"

# --- Step 2: Run CrossAffinity inference on CPU ---
cd ${CROSS_DIR}
python inference.py \
    --filepath ${INPUT_CSV} \
    --output ${OUTPUT_CSV} \
    --batch_size 8 \
    --esm2_device cpu \
    --cross_affinity_device cpu \
    --num_workers 2

# --- Step 3: Merge predictions back with IDs ---
python -c "
import pandas as pd

gt = pd.read_csv('${PROTAFF_DIR}/data/test/test_adaptyv.csv')
pred = pd.read_csv('${OUTPUT_CSV}')

# CrossAffinity output has 'Part 1', 'Part 2', per-fold columns, and 'pKd'
# Match rows by sequence pairs (order is preserved)
gt['pKd'] = pred['pKd'].values
gt[['id', 'binder_sequence', 'target_sequence', 'log_Aff', 'pKd']].to_csv(
    '${PROTAFF_DIR}/results/crossaffinity_adaptyv_predictions.csv', index=False)
print('Saved merged predictions with IDs')
print(gt[['id', 'log_Aff', 'pKd']].head())
"

echo "Done. Results at: ${PROTAFF_DIR}/results/crossaffinity_adaptyv_predictions.csv"
