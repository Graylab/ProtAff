#!/bin/bash
#SBATCH --partition=a100
#SBATCH --account=jgray21_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=2:00:00
#SBATCH --output=slogs/%j.out
#SBATCH --job-name=champloo_mint
set -euo pipefail

# Run MINT inference on champloo all-vs-all pairs
PROTAFF_DIR=/scratch16/jgray21/lchu11/projects/ProtAff
MINT_DIR=/scratch16/jgray21/lchu11/projects/mint
INPUT_CSV=${PROTAFF_DIR}/data/sources/ab_ag_champloo/champloo_allvsall.csv
OUTPUT_CSV=${PROTAFF_DIR}/results/champloo/mint_predictions.csv

if [ ! -f "${INPUT_CSV}" ]; then
    echo "ERROR: ${INPUT_CSV} not found. Run prepare_champloo_pairs.py first."
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT_CSV}")"

echo "=== MINT inference on champloo benchmark ==="
echo "Input: ${INPUT_CSV}"
echo "Output: ${OUTPUT_CSV}"

cd "${MINT_DIR}"

python -c "
import functools, torch
# MINT checkpoints contain argparse.Namespace; PyTorch 2.6+ rejects them
# with weights_only=True (the new default). Patch torch.load to allow it.
_orig_load = torch.load
@functools.wraps(_orig_load)
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

import pandas as pd
from tqdm import tqdm
from mint.helpers.extract import load_config, CSVDataset, CollateFn, MINTWrapper
from mint.helpers.predict import SimpleMLP

# Configuration
cfg = load_config('data/esm2_t33_650M_UR50D.json')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
checkpoint_path = 'mint.ckpt'
mlp_checkpoint_path = 'bernett_mlp.pth'

csv_file = '${INPUT_CSV}'
seq1 = 'binder_sequence'
seq2 = 'target_sequence'

# Load data
df = pd.read_csv(csv_file)
dataset = CSVDataset(csv_file, seq1, seq2)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=CollateFn(), shuffle=False)

# Initialize models
wrapper = MINTWrapper(cfg, checkpoint_path, sep_chains=True, device=device)
model = SimpleMLP()
mlp_checkpoint = _orig_load(mlp_checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(mlp_checkpoint)
model.to(device)
model.eval()

# Inference
all_predictions = []
print(f'Starting MINT inference on {len(df)} pairs...')
with torch.no_grad():
    for chains, chain_ids in tqdm(loader):
        chains = chains.to(device)
        chain_ids = chain_ids.to(device)
        embeddings = wrapper(chains, chain_ids)
        raw_preds = model(embeddings)
        probs = torch.sigmoid(raw_preds)
        all_predictions.append(probs.item())

# Save
df['interaction_prob'] = all_predictions
df.to_csv('${OUTPUT_CSV}', index=False)
print(f'Done! {len(all_predictions)} predictions saved to ${OUTPUT_CSV}')
"

echo "Done. Results at: ${OUTPUT_CSV}"
