#!/bin/bash
MODEL="esm_interaction_map"
python src/inference.py model_path=$1 input_csv=data/test/test_adaptyv.csv model=$MODEL

