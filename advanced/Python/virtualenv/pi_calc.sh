#!/bin/bash -l
#SBATCH --job-name=pi_calc
#SBATCH --output=pi_calc.log
#
#SBATCH --ntasks=9
#SBATCH --time=0-00:35:00
#SBATCH -p batch
#SBATCH --qos=qos-batch
#
#SBATCH --array=2-10

python pi_calc.py $SLURM_ARRAY_TASK_ID
