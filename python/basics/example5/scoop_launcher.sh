#!/bin/bash -l
#SBATCH --job-name=scoop
#SBATCH --output=scoop_%j.log
#
#SBATCH -N 2
#SBATCH -n 56
#SBATCH -c 1
#SBATCH --time=0-00:35:00
#
#SBATCH -p batch
#SBATCH --qos=qos-batch
#
#SBATCH --exclusive
#SBATCH -d singleton
#SBATCH -m cyclic
#SBATCH -C skylake
#
#SBATCH --array=1-55
#
# Ensure process affinity is disabled
export SLURM_CPU_BIND=none
hosts=$(srun bash -c hostname)
echo $hosts
NB_WORKERS=$SLURM_ARRAY_TASK_ID
echo $NB_WORKERS
python -m scoop --hosts $hosts -n $NB_WORKERS pi_calc.py $NB_WORKERS

