#!/bin/bash -l
#SBATCH -N 2
#SBATCH --ntasks-per-node=3
#SBATCH --time=00:05:00
#SBATCH -p batch
#SBATCH --reservation=hpcschool

module purge
module load swenv/default-env/latest
module load toolchain/intel/2019a
module load tools/VTune/2019_update4
module load vis/GTK+/3.24.8-GCCcore-8.2.0

srun -n $SLURM_NTASKS amplxe-cl -collect uarch-exploration -r vtune_mpi -- ./a.out
