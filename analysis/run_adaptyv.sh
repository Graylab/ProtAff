#!/bin/bash
MODEL="esm_cross_attn"
python src/inference.py model_path=$1 input_csv=data/test/test_adaptyv.csv model=$MODEL

