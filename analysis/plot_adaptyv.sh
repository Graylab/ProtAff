#!/bin/bash

python analysis/plot_scatter.py $1
python analysis/plot_correlation.py $1
python analysis/plot_enrichment_bars.py $1 
python analysis/plot_roc_success_rate.py $1

