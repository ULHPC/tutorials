#!/bin/bash -l
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128


# Increase the user process limit.
ulimit -u 10000

echo "Spawning ${SLURM_NTASKS_PER_NODE} parallel worker on ${SLURM_NNODES} nodes"
echo "Nodes: ${SLURM_NODELIST}"
echo "Each parallel worker can execute ${SLURM_CPUS_PER_TASK} independant tasks" 
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export TIMESTAMP=$(date +"%Y%m%dT%H%M%S")
srun --no-kill --wait=0 $*
