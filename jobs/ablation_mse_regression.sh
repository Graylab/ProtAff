#!/bin/bash
#SBATCH --partition=a100
#SBATCH --account=jgray21_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=24:00:00
#SBATCH --output=slogs/%j.out

# Ablation: MSE regression instead of margin ranking
srun python src/train.py --config-name affinity \
    model="esm_binding" \
    model.use_lora=true \
    data.base_csv="data/combined/base_cleaned.csv" \
    data.lookup_csv="data/combined/lookup_varonly.csv" \
    training.seed=42 \
    training.strategy="ddp"
