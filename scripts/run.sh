#!/bin/bash

python scripts/inference.py model_path=$1 input_csv=data/test/test_abrank.csv
python scripts/inference.py model_path=$1 input_csv=data/test/test_aintibody.csv
python scripts/inference.py model_path=$1 input_csv=data/test/test_adaptyv.csv
python scripts/analyze_results.py --csv inference_results/test_abrank/predictions.csv
python scripts/analyze_results.py --csv inference_results/test_aintibody/predictions.csv
python scripts/analyze_results.py --csv inference_results/test_adaptyv/predictions.csv

