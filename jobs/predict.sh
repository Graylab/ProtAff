#!/bin/bash
#SBATCH --partition=a100
#SBATCH --account=jgray21_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=1:00:00
#SBATCH --output=slogs/%j.out

python tools/predict.py \
    --model_dir $1 \
    --input_csv data/sources/az_collab/predict_input_untrim.csv \
    --output_csv data/sources/az_collab/predict_output_untrim.csv
