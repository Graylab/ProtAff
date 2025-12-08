#!/bin/bash
MODEL="esm2_t30_150M_UR50D"
python scripts/analyze_results.py --csv inference_results/$MODEL/test_abrank/predictions.csv
python scripts/analyze_results.py --csv inference_results/$MODEL/test_aintibody/predictions.csv
python scripts/analyze_results.py --csv inference_results/$MODEL/test_adaptyv/predictions.csv

