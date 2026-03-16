#!/bin/bash
#SBATCH --partition=a100
#SBATCH --account=jgray21_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=30:00
#SBATCH --output=slogs/%j.out

MODEL="esm_binding"

python src/extract_attention.py \
    model_path=$1 \
    input_csv=data/test/test_adaptyv.csv \
    model=$MODEL \
    output_dir=$2
