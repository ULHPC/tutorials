#!/bin/bash -l
#SBATCH -N 2
#SBATCH --ntasks-per-node=3
#SBATCH --time=00:05:00
#SBATCH -p batch
#SBATCH --reservation=hpcschool

module purge
module load swenv/default-env/latest
module load toolchain/intel/2019a
module load perf/Advisor/2019_update4

srun -n 6 advixe-cl --collect survey --project-dir result -- ./a.out

