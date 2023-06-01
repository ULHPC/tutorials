#!/bin/bash -l
#SBATCH --job-name=mpi_job_test      # Job name
#SBATCH --cpus-per-task=1            # Number of cores per MPI task
#SBATCH --nodes=2                    # Maximum number of nodes to be allocated
#SBATCH --ntasks-per-node=128        # Maximum number of tasks on each node
#SBATCH --output=mpi_test_%j.log     # Path to the standard output and error files relative to the working directory
#SBATCH --exclusive
#SBATCH -p batch

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
OSU_VERSION="7.1-1"
OSU_ARCHIVE="osu-micro-benchmarks-${OSU_VERSION}.tar.gz"
OSU_URL="https://mvapich.cse.ohio-state.edu/download/mvapich/${OSU_ARCHIVE}"

if [[ ! -f ${OSU_ARCHIVE} ]];then 
    wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.1-1.tar.gz
    tar -xvf ${OSU_ARCHIVE} 
fi

# We use the hash since we could have different variants of the same openmpi version
# Adapt with your hash version
spack load /xgcbqft

# cd into the extracted folder 
cd ${OSU_ARCHIVE//.tar.gz/}

# configure
./configure CC=$(which mpicc) CXX=$(which mpicxx)
make
cd ..

srun  ${OSU_ARCHIVE//.tar.gz/}/c/mpi/collective/blocking/osu_alltoall
