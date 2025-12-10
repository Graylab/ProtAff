#!/bin/bash
MODEL="alphabio_esm2_150M_from_scratch"

# plot scatter
python analysis/plot_scatter.py --csv inference_results/$MODEL/test_abrank/predictions.csv
python analysis/plot_scatter.py --csv inference_results/$MODEL/test_aintibody/predictions.csv
python analysis/plot_scatter.py --csv inference_results/$MODEL/test_adaptyv/predictions.csv

