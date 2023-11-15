#!/bin/sh -l
#SBATCH -c 60
#SBATCH -N 2
#SBATCH -t 5
#SBATCH --export=ALL

echo "Hello I am ${HOSTNAME}. I received the input=$SLURM_ARRAY_TASK_ID. It is $(date)."

