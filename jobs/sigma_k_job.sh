#!/bin/bash -l

#SBATCH -A snic2020-15-133
#SBATCH -p core
#SBATCH -n 8
#SBATCH -t 0-12:00:00
#SBATCH -J sigma_k_plot_job
#SBATCH --mail-type='ALL'
#SBATCH --mail-user=Zack.Kite.2016@student.uu.se

pyenv global 3.7.3
cd /home/zaki2016/ml-calabi-yau-metric
python3 sigma_measure.py -k 4 -N 500000