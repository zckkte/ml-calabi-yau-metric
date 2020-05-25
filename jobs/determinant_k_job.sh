#!/bin/bash -l 
 
#SBATCH -A snic2020-15-133 
#SBATCH -p core 
#SBATCH -n 4 
#SBATCH -t 0-3:25:00 
#SBATCH -J determinant_k_job 
#SBATCH --mail-type='ALL' 
#SBATCH --mail-user=Zack.Kite.2016@student.uu.se 
 
cd /home/zaki2016/ml-calabi-yau-metric 
pyenv global 3.7.3 
python3 donaldson.py -k 5 