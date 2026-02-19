# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProtAff is a protein binding affinity prediction framework built on ESM2 (protein language model) with LoRA fine-tuning. It supports both regression and pairwise ranking tasks across two domains: binding affinity and protein-protein interactions (PPI).

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

Four tasks configured via `--config-name` in `configs/`:

| Config | Task | Loss | Key Metric |
|--------|------|------|------------|
| `affinity` | Affinity regression | MSE | Spearman |
| `pair_affinity` | Affinity ranking | Margin ranking | Spearman |
| `ppi` | PPI regression | MSE | Spearman |
| `pair_ppi` | PPI ranking | Margin ranking | Spearman |

Each task has a config `configs/<task>.yaml` and a dataset `src/datasets/dataset_<task>.py`. Lightning modules are unified: `RegressionModule` (for affinity, ppi) and `RankingModule` (for pair_affinity, pair_ppi), both inheriting from `BaseModule` in `src/lightning/base_module.py`. The factory `TASK_MAP` in `src/train.py` dispatches based on `task_name`.

### Model Architectures

Three ESM2-based architectures registered in `src/models/__init__.py` (selected via `cfg.model.arch`):

- **`concat`** (`ESMConcatModel`): Concatenates binder+target as `[CLS] Binder [EOS] Target [EOS]`, single self-attention pass. Max 2048 tokens.
- **`cross_attn`** (`ESMCrossAttnModel`): Separate encoding, uni-directional cross-attention (binder queries target). Max 1024 tokens each.
- **`bi_cross_attn`** (`ESMBiCrossAttnModel`): Bidirectional cross-attention, concatenated pooling. Max 1024 tokens each.

All models: ESM2 backbone (`facebook/esm2_t33_650M_UR50D`) → LoRA on last N layers → projection (1280→256) → cross-attention layers → score head.

Model configs live in `configs/model/` (e.g., `esm_cross_attn.yaml`).

### Data Pipeline

Shared collators in `src/datasets/collators.py` handle architecture-specific tokenization:
- `ConcatCollator` / `CrossAttnCollator` for regression tasks
- `PairwiseConcatCollator` / `PairwiseCrossAttnCollator` for ranking tasks (produces better/worse pairs)
- `RegressionTestCollator` / `BinaryClassificationCollator` for test evaluation
- `select_collator(arch, tokenizer, max_length, mode)` factory function

Shared test datasets in `src/datasets/test_datasets.py`: `TestRegressionDataset`, `BinaryClassificationTestDataset`.

Data splitting uses group-based strategy (`split_strategy: "group"`) to keep entire targets unseen in validation. Cluster-balanced sampling controlled by `balance_clusters` and `balance_power`.

### Transfer Learning

Two-phase training: pretrain on PPI, then fine-tune on affinity. Set `pretrained_ckpt_path` in config to load Phase 1 LoRA weights.

### Configuration

Hydra YAML configs in `configs/`. Key override patterns:
- `model=esm_cross_attn` selects model config from `configs/model/`
- `training.loss_type=soft_margin` switches loss function
- `pretrained_ckpt_path=path/to/saved_model` enables transfer learning
- `resume_checkpoint_path=path/to/last.ckpt` resumes from crash

Outputs go to `outputs/{task_name}/{date}/{time}/` containing checkpoints, saved_model (LoRA adapter), and wandb logs.

### Logging and Evaluation

W&B logging is configured per-task in `training.wandb.project`. Custom callbacks in `src/train.py`:
- `BestModelSaver`: exports best LoRA adapter to `saved_model/`
- `TestEveryValidationCallback`: regression test on held-out data each validation
- `BinaryTestCallback`: per-target AUC/AP on binary binder classification
