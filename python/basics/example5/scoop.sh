#!/bin/bash -l
#SBATCH --job-name=scoop
#SBATCH --output=scoop.log
#SBATCH --open-mode=append
#
#SBATCH -N 2
#SBATCH --ntasks=56
#SBATCH --time=0-00:35:00
#SBATCH -p batch
#SBATCH --qos=qos-batch
#
#SBATCH --array=1-56%1
#SBATCH -d singleton
#SBATCH --exclusive

python -m scoop -n $SLURM_ARRAY_TASK_ID pi_calc.py $SLURM_ARRAY_TASK_ID
