#!/bin/bash -l
#SBATCH --job-name=scoop
#SBATCH --output=scoop_%j.log
#
#SBATCH -N 2
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:35:00
#
#SBATCH -p batch
#SBATCH --qos=qos-batch
#
#SBATCH --exclusive
#SBATCH -m cyclic
#SBATCH -C skylake
#
# Ensure process affinity is disabled
export SLURM_CPU_BIND=none
hosts=$(srun bash -c hostname)
echo $hosts
echo $((SLURM_NTASKS-1))
python -m scoop --hosts $hosts -n $((SLURM_NTASKS-1)) pi_calc.py $((SLURM_NTASKS-1))
