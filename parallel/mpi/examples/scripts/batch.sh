#!/bin/bash -l
#SBATCH -N 2
#SBATCH --ntasks-per-node=7
#SBATCH --time=00:15:00
#SBATCH -p batch
#SBATCH --reservation=hpcschool

module purge
module load swenv/default-env/latest
module load toolchain/intel/2019a

srun -n $SLURM_NTASKS ./a.out
