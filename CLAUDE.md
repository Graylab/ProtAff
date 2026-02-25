# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProtAff is a protein binding affinity prediction framework built on ESM2 (protein language model) with LoRA fine-tuning. It supports both regression and pairwise ranking tasks for binding affinity prediction.

## Commands

### Training
```bash
# Direct (Hydra selects config by --config-name)
python src/train.py --config-name pair_affinity

# With overrides
python src/train.py --config-name pair_affinity model=esm_bi_cross_attn training.learning_rate=1e-4

# Distributed via SLURM (see jobs/ for examples)
srun -n 4 python src/train.py --config-name pair_affinity
```

### Inference
```bash
python src/inference.py model_path="outputs/pair_affinity/2026-01-01/12-00-00/saved_model" input_csv="data/test/test.csv"
```

### Zero-shot baseline
```bash
python src/zero_shot.py input_csv="data/test/test.csv"
```

There is no test suite, linter configuration, or requirements.txt. Dependencies include: torch, pytorch-lightning, transformers, peft, hydra-core, omegaconf, torchmetrics, scikit-learn, pandas, numpy, tqdm, wandb.

## Architecture

### Task System

Two tasks configured via `--config-name` in `configs/`:

| Config | Task | Loss | Key Metric |
|--------|------|------|------------|
| `affinity` | Affinity regression | MSE | Spearman |
| `pair_affinity` | Affinity ranking | Margin ranking | Spearman |

Each task has a config `configs/<task>.yaml` and a dataset `src/datasets/dataset_<task>.py`. Lightning modules are unified: `RegressionModule` (for affinity) and `RankingModule` (for pair_affinity), both inheriting from `BaseModule` in `src/lightning/base_module.py`. The factory `TASK_MAP` in `src/train.py` dispatches based on `task_name`.

### Model Architecture

Single ESM2-based architecture (`ESMBindingModel` in `src/models/esm_model.py`): separate encoding of binder and target, uni-directional cross-attention (binder queries target), attention pooling, and affinity score head.

ESM2 backbone (`facebook/esm2_t33_650M_UR50D`) → LoRA on last N layers → projection (1280→256) → cross-attention layers → affinity head. Max 1024 tokens each.

Model config: `configs/model/esm_binding.yaml`.

### Data Pipeline

Collators in `src/datasets/collators.py` handle tokenization (separate binder/target encoding):
- `CrossAttnCollator` for regression tasks
- `PairwiseCrossAttnCollator` for ranking tasks (produces better/worse pairs)
- `RegressionTestCollator` / `BinaryClassificationCollator` / `InferenceCollator` for evaluation
- `select_collator(tokenizer, max_length, mode)` factory function

Shared test datasets in `src/datasets/test_datasets.py`: `TestRegressionDataset`, `BinaryClassificationTestDataset`.

Data splitting uses group-based strategy (`split_strategy: "group"`) to keep entire targets unseen in validation. Cluster-balanced sampling controlled by `balance_clusters` and `balance_power`.

### Transfer Learning

Set `pretrained_ckpt_path` in config to load pretrained LoRA weights for fine-tuning.

### Configuration

Hydra YAML configs in `configs/`. Key override patterns:
- `training.loss_type=soft_margin` switches loss function
- `pretrained_ckpt_path=path/to/saved_model` enables transfer learning
- `resume_checkpoint_path=path/to/last.ckpt` resumes from crash

Outputs go to `outputs/{task_name}/{date}/{time}/` containing checkpoints, saved_model (LoRA adapter), and wandb logs.

### Logging and Evaluation

W&B logging is configured per-task in `training.wandb.project`. Custom callbacks in `src/train.py`:
- `BestModelSaver`: exports best LoRA adapter to `saved_model/`
- `TestEveryValidationCallback`: regression test on held-out data each validation
- `BinaryTestCallback`: per-target AUC/AP on binary binder classification
