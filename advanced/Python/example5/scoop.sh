#!/bin/bash -l
#SBATCH --job-name=scoop
#SBATCH --output=scoop.log
#SBATCH --open-mode=append
#
#SBATCH -N 2
#SBATCH --ntasks=50
#SBATCH --time=0-00:35:00
#SBATCH -p batch
#SBATCH --qos=qos-batch
#
#SBATCH --array=1-50%1
#SBATCH -d singleton

python -m scoop -n $SLURM_ARRAY_TASK_ID pi_calc.py $SLURM_ARRAY_TASK_ID
