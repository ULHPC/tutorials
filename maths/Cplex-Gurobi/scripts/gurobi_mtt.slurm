#!/bin/bash -l
#SBATCH -J Multi-threaded_gurobi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=0-01:00:00
#SBATCH -p batch
#SBATCH --qos=normal

# Explicit cpus-per-task to srun
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# Load Gurobi 
module load math/Gurobi/8.1.1-intel-2018a-Python-3.6.4

# Some variable
MPS_FILE=$1
RES_FILE=$2

# Gurobi will access use the required number of thread
gurobi_cl Threads=${SLURM_CPUS_PER_TASK} ResultFile="${RES_FILE}.sol" ${MPS_FILE}
