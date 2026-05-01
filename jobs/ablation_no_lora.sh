#!/bin/bash
#SBATCH --partition=a100
#SBATCH --account=jgray21_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=12
#SBATCH --time=24:00:00
#SBATCH --output=slogs/%j.out

# Ablation: Frozen ESM2 (no LoRA fine-tuning)
srun python src/train.py --config-name pair_affinity \
    model="esm_binding" \
    model.use_lora=false \
    data.base_csv="data/combined/base_cleaned.csv" \
    data.lookup_csv="data/combined/lookup_varonly.csv" \
    training.seed=42 \
    training.strategy="ddp"
