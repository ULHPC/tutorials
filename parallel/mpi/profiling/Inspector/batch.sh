#!/bin/bash -l
#SBATCH -N 2
#SBATCH --ntasks-per-node=7
#SBATCH --time=00:05:00
#SBATCH -p batch
#SBATCH -A ulhpc

module purge
module load swenv/default-env/latest
module load toolchain/intel/2019a
module load tools/Inspector/2019_update4

srun -n $SLURM_NTASKS inspxe-cl -collect=ti2 -r result ./a.out

# on the login node
# mpirun-np 14 inspxe-cl -collect=ti2 -r result ./a.out
