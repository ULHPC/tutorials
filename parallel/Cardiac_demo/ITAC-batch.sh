#!/bin/bash -l
#SBATCH -N 2
#SBATCH --ntasks-per-node=3
#SBATCH --time=00:15:00
#SBATCH -p batch
#SBATCH --reservation=hpcschool

module purge
module load toolchain/intel/2019a
module load tools/itac/2019.4.036
module load tools/VTune/2019_update4
module load vis/GTK+/3.24.8-GCCcore-8.2.0

srun -n 6 ./a.out
