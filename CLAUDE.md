# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProtAff is a protein binding affinity prediction framework built on ESM2 (protein language model) with LoRA fine-tuning. It supports both regression and pairwise ranking tasks for binding affinity prediction.

**Key convention: lower predicted affinity = stronger binder** (natural Kd scale). The ranking module trains `scores_better < scores_worse`.

## Commands

### Training
```bash
# Direct (Hydra selects config by --config-name)
python src/train.py --config-name pair_affinity

# With overrides
python src/train.py --config-name pair_affinity training.learning_rate=1e-4

# Distributed via SLURM (see jobs/ for examples)
srun -n 4 python src/train.py --config-name pair_affinity
```

### Inference
```bash
python src/inference.py model_path="outputs/pair_affinity/2026-01-01/12-00-00/saved_model" input_csv="data/test/test.csv"

# Standalone prediction script (no Hydra dependency)
python src/predict.py --model_dir /path/to/saved_model --input_csv data.csv --output_csv results.csv
# Use --target_seq to apply the same target to all binders
```

### Attention extraction
```bash
python src/extract_attention.py model_path="path/to/saved_model" input_csv="data.csv"
# Saves .npz with per-layer attention weights and pool weights
```

### Zero-shot baseline
```bash
python src/zero_shot.py input_csv="data/test/test.csv"
```

## Directory Structure

```
scripts/          # Shell scripts (.sh only)
analysis/         # All figure generation and analysis scripts
  paper/          # Paper figure outputs
  slides/         # Slide figure outputs
data/
  sources/        # Raw dataset sources (abrank, alphabio, asd, az_collab, binder, NIV, merged)
  structures/     # Structure prediction outputs (af3, boltz2)
  pipelines/      # Data processing pipelines (cluster_pipeline, leakage_pipeline, process.py)
  combined/       # Primary training data
  extended/       # Extended training data
  test/           # Test datasets
  splits/         # Train/val split files
src/              # Core source code (models, datasets, lightning modules, training)
configs/          # Hydra YAML configs
jobs/             # SLURM job scripts
```

There is no test suite, linter configuration, or requirements.txt. Dependencies include: torch, pytorch-lightning, transformers, peft, hydra-core, omegaconf, torchmetrics, scikit-learn, pandas, numpy, tqdm, wandb.

## Architecture

### Task System

Two tasks configured via `--config-name` in `configs/`:

| Config | Task | Loss Options | Key Metric |
|--------|------|-------------|------------|
| `affinity` | Affinity regression | `mse` (default), `smooth_l1`, `huber`, `bce`, `focal` | Spearman |
| `pair_affinity` | Affinity ranking | `margin` (default), `soft_margin`, `bce`, `contrastive`, `lambdarank` | Spearman |

Each task has a config `configs/<task>.yaml` and a dataset `src/datasets/dataset_<task>.py`. Lightning modules: `RegressionModule` (for affinity) and `RankingModule` (for pair_affinity), both inheriting from `BaseModule` in `src/lightning/base_module.py`. The factory `TASK_MAP` in `src/train.py` dispatches based on `task_name` (default: `"affinity"`).

### Model Architecture

Single ESM2-based architecture (`ESMBindingModel` in `src/models/esm_model.py`): separate encoding of binder and target, uni-directional cross-attention (binder queries target), attention pooling, and affinity score head.

ESM2 backbone (`facebook/esm2_t33_650M_UR50D`) → LoRA on last N layers → projection (1280→256) → cross-attention layers → affinity head. Max 1024 tokens each.

Forward signature: `forward(binder_ids, binder_mask, target_ids, target_mask, return_attn=False, **kwargs)`. With `return_attn=True`, returns `(scores, attn_weights_list, pool_weights)`.

Custom layers saved on export: `input_proj`, `input_norm`, `cross_layers`, `pool`, `head_affinity`.

Model config: `configs/model/esm_binding.yaml`. Baseline (frozen ESM2, no LoRA) mode: set `use_lora: false`; exports custom layers only to `baseline_model.pt`.

### Data Pipeline

**Data format:** `base_csv` contains `[binder_id, target_id, log_Aff]`. A `lookup_csv` maps `(type, id)` to sequences, keyed as `"binder_" + binder_id` or `"target_" + target_id`.

Collators in `src/datasets/collators.py` handle tokenization (separate binder/target encoding):
- `CrossAttnCollator` for regression tasks → `{binder_ids, binder_mask, target_ids, target_mask, reg_labels}`
- `PairwiseCrossAttnCollator` for ranking tasks → `{better_*, worse_*, delta, lambda_weight}`
- `BinaryClassificationCollator` / `InferenceCollator` for evaluation
- `select_collator(tokenizer, max_length, mode)` factory function

Shared test datasets in `src/datasets/test_datasets.py`: `TestRegressionDataset`, `BinaryClassificationTestDataset`.

**Label normalization:** Labels are normalized by `(raw - mean) / (std + 1e-8)`. Validation always uses train set statistics (`provided_stats`) to prevent label leakage.

**Splitting:** Group-based strategy (`split_strategy: "group"`) keeps entire targets unseen in validation. Singletons always go to train. Also available: `"random"`, `"within_group"`. Cluster-balanced sampling controlled by `balance_clusters` and `balance_power`.

**Pair generation (pair_affinity):** Intra-target pairs filtered by `[min_margin, max_margin]` on log_Aff delta. Inter-target pairs sample across targets. LambdaRank weights (`|ΔNDCG|`) computed only for intra-target pairs. Sampler uses sqrt-geometric-mean weighting across `(source, target_id)` groups.

### Transfer Learning

Set `pretrained_ckpt_path` in config to load pretrained LoRA weights for fine-tuning.

### Configuration

Hydra YAML configs in `configs/`. Key override patterns:
- `training.loss_type=soft_margin` switches loss function
- `pretrained_ckpt_path=path/to/saved_model` enables transfer learning
- `resume_checkpoint_path=path/to/last.ckpt` resumes from crash

Outputs go to `outputs/{task_name}/{date}/{time}/` containing checkpoints, saved_model (LoRA adapter), and wandb logs.

### Logging and Evaluation

W&B logging is configured per-task in `training.wandb.project`. Custom callbacks in `src/callbacks.py`:
- `BestModelSaver`: exports best LoRA adapter (or baseline weights) to `saved_model/`
- `TestEveryValidationCallback`: regression test on held-out data each validation
- `BinaryTestCallback`: per-target AUC/AP on binary binder classification

### Multi-GPU

DDP strategy supported. Uses `trainer.is_global_zero` checks for logging/saving and all-gather for metric aggregation across processes.
