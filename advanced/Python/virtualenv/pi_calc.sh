#!/bin/bash -l
#SBATCH --job-name=pi_calc
#SBATCH --output=pi_calc.log
#
#SBATCH -N 2
#SBATCH --ntasks==1
#SBATCH --time=0-00:35:00
#SBATCH -p batch
#SBATCH --qos=qos-batch
#
#SBATCH --array=100,1000,10000,100000,1000000,10000000

python pi_calc.py $SLURM_ARRAY_TASK_ID
