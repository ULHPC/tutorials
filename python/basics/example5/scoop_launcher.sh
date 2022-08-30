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
#SBATCH --qos=normal
#
#SBATCH --exclusive
#SBATCH -d singleton
#SBATCH -m cyclic

# only for Iris
##SBATCH -C skylake
#
# Ensure process affinity is disabled
export SLURM_CPU_BIND=none
hosts=$(srun bash -c hostname)
echo $hosts

# change number of workers here
NB_WORKERS=20

echo $NB_WORKERS
python -m scoop --hosts $hosts -n $NB_WORKERS pi_calc.py $NB_WORKERS

