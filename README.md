# ProtAff

Protein binding affinity prediction with ESM2 + LoRA cross-attention.

> **Convention:** lower predicted affinity = stronger binder (natural Kd scale).

## Pretrained model

The recommended checkpoint for general use is `weights/model_margin_weighted/`
(LoRA adapter trained with weighted margin ranking on log-affinity pairs).

```
weights/model_margin_weighted/
├── adapter_config.json        # PEFT/LoRA config
├── adapter_model.safetensors  # LoRA weights
├── tokenizer_config.json      # ESM2 tokenizer
├── special_tokens_map.json
├── vocab.txt
└── best_model_metadata.txt
```

## Installation

```bash
git clone https://github.com/<your-org>/ProtAff.git
cd ProtAff
```

Pick **one** of the following install paths.

**Option 1 — pip (recommended for inference, ~2 min, ~3 GB):**

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Option 2 — conda (minimal, equivalent to option 1):**

```bash
conda env create -f environment.yml   # creates env named "protaff"
conda activate protaff
```

**Option 3 — full training/analysis env** (lightning, wandb, anarci,
foldseek, biotite, …): `conda env create -f environment-full.yml`. Only
needed if you plan to retrain or run the analysis scripts.

Minimum requirements for inference: Python ≥ 3.10, PyTorch ≥ 2.1,
`transformers`, `peft`, `safetensors`, `omegaconf`, `pandas`, `tqdm`,
`pyyaml`. A GPU is strongly recommended; CPU inference works but is slow
(~1 s/sequence on CPU with ESM2-650M).

The first run downloads `facebook/esm2_t33_650M_UR50D` (~2.5 GB) from
HuggingFace Hub.

## Quick start — run inference

`tools/predict.py` is a standalone script (no Hydra) that loads the LoRA
adapter on top of ESM2 and writes per-row predictions.

### 1. Prepare an input CSV

Your CSV needs a binder column and a target column. Accepted names:

| Column  | Accepted headers                                                |
|---------|-----------------------------------------------------------------|
| binder  | `binder_sequence`, `binder_seq`, `binder`, `seq_1`, `heavy_chain`, `cdr3` |
| target  | `target_sequence`, `target_seq`, `target`, `seq_2`, `antigen`   |

Example `inputs.csv`:

```csv
id,binder_sequence,target_sequence
binder_A,EVQLVQSGAEVKK...,LEEKKVCQGTSNK...
binder_B,EVKLEESGGGLVQ...,LEEKKVCQGTSNK...
```

### 2. Run prediction

```bash
python tools/predict.py \
    --model_dir weights/model_margin_weighted \
    --input_csv inputs.csv \
    --output_csv predictions.csv
```

The output is your input CSV with one extra column, `predicted_affinity`
(lower = stronger binder).

### 3. Common options

Apply the same target to every row (skip the target column in the CSV):

```bash
python tools/predict.py \
    --model_dir weights/model_margin_weighted \
    --input_csv binders_only.csv \
    --output_csv predictions.csv \
    --target_seq MKWVTFISLLFLFSSAYS...
```

Tune throughput / memory:

```bash
python tools/predict.py \
    --model_dir weights/model_margin_weighted \
    --input_csv inputs.csv \
    --output_csv predictions.csv \
    --batch_size 16 \
    --num_workers 4
```

| Flag           | Default | Notes                                                 |
|----------------|---------|-------------------------------------------------------|
| `--model_dir`  | —       | Path to the LoRA adapter folder.                      |
| `--input_csv`  | —       | Input CSV with binder (and optional target) columns.  |
| `--output_csv` | —       | Where to write predictions.                           |
| `--batch_size` | 32      | Lower if you hit OOM.                                 |
| `--num_workers`| 0       | DataLoader workers.                                   |
| `--target_seq` | None    | Apply one target sequence to all rows.                |
| `--max_length` | 1024    | Token cap per chain (binder and target each).         |

## End-to-end example

```bash
# 1. Activate environment
source .venv/bin/activate           # or: conda activate protaff

# 2. Predict on the bundled NIV test set
python tools/predict.py \
    --model_dir weights/model_margin_weighted \
    --input_csv data/test/test_niv.csv \
    --output_csv predictions_niv.csv

# 3. Rank: lowest predicted_affinity = strongest binder
python -c "import pandas as pd; \
df = pd.read_csv('predictions_niv.csv'); \
print(df.nsmallest(5, 'predicted_affinity')[['id','predicted_affinity']])"
```

## Interpreting the score

`predicted_affinity` is in normalized log-Kd units (lower = stronger).
The model was trained with a pairwise margin loss, so absolute values are
**only meaningful as a ranking**: compare scores across binders for the
same target rather than treating any single value as a calibrated Kd.

## Troubleshooting

- **`OSError: ... adapter_model.safetensors not found`** — point `--model_dir`
  at the folder that contains `adapter_config.json`, not at any single file.
- **CUDA OOM** — drop `--batch_size` (try 8 or 4) or run on CPU. Sequences
  are truncated to `max_length=1024` tokens per chain.
- **`Could not find binder column in CSV`** — rename your column to one of
  the accepted headers above.
- **First run is slow** — the ESM2-650M backbone (~2.5 GB) is downloaded
  once and cached under `~/.cache/huggingface/`.

## Citation

If you use ProtAff or these weights, please cite the accompanying paper
(see the repository for current citation details).
