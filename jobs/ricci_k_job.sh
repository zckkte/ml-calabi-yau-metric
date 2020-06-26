#!/bin/bash -l

#SBATCH -A snic2020-15-133
#SBATCH -p core
#SBATCH -n 16
#SBATCH -t 0-72:00:00
#SBATCH -J ricci_k_plot_job
#SBATCH --mail-type='ALL'
#SBATCH --mail-user=Zack.Kite.2016@student.uu.se

pyenv global 3.7.3
cd /home/zaki2016/ml-calabi-yau-metric
python3 global_ricci_measure.py -k 4 -N 10000