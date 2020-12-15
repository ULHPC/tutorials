#!/bin/bash -l
#SBATCH -N 2
#SBATCH --ntasks-per-node=3
#SBATCH --time=00:15:00
#SBATCH -p batch
#SBATCH --reservation=hpcschool

module purge
module load swenv/default-env/latest
module load toolchain/foss/2019a
module load mpi/OpenMPI/3.1.4-GCC-8.2.0-2.31.1
module load tools/ArmForge/19.1

unset I_MPI_PMI_LIBRARY
map --profile mpirun -np 6 ./a.out

