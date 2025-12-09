#!/bin/bash
MODEL="esm2_t30_150M_UR50D"

# plot scatter
python analysis/plot_scatter.py --csv inference_results/$MODEL/test_abrank/predictions.csv
python analysis/plot_scatter.py --csv inference_results/$MODEL/test_aintibody/predictions.csv
python analysis/plot_scatter.py --csv inference_results/$MODEL/test_adaptyv/predictions.csv

# plot correlation
#python analysis/plot_correlation.py 

# plot roc success rate
#python analysis/plot_roc_success_rate.py

