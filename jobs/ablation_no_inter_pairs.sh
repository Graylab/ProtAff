#!/bin/bash
#SBATCH --partition=ica100,a100
#SBATCH --account=jgray21_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=12
#SBATCH --time=36:00:00
#SBATCH --output=slogs/%j.out

# Ablation: No inter-target pairs (intra-target only)
srun python src/train.py --config-name pair_affinity \
    model="esm_binding" \
    model.use_lora=true \
    data.base_csv="data/combined/base_cleaned.csv" \
    data.lookup_csv="data/combined/lookup_varonly.csv" \
    data.inter_samples_per_target=0 \
    data.inter_pps=0 \
    training.seed=42 \
    training.strategy="ddp"
