#!/bin/bash -l
#SBATCH --job-name=scooplauncher
#SBATCH --output=scooplauncher.log
#
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --time=0-00:35:00
#SBATCH -p batch
#SBATCH --qos=qos-batch
#
#SBATCH --array=2-56

sbatch --ntasks=$SLURM_ARRAY_TASK_ID scoop_launcher.sh
