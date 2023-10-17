#!/bin/bash -l
#SBATCH -J Distrbuted_cplex
#SBATCH --nodes=2
#SBATCH --tasks=14
#SBATCH --cpus-per-task=2
#SBATCH --time=0-01:00:00
#SBATCH -p batch
#SBATCH --qos=normal

# Explicit cpus-per-task to srun
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

module load math/CPLEX/12.8-foss-2018a

# Some variables
MPS_FILE=$1
RES_FILE=$2
CPLEX_COMMAND_SCRIPT="command_job${SLURM_JOBID}.lst"



# Create cplex command script
cat << EOF > ${CPLEX_COMMAND_SCRIPT}
set distmip config mpi
set threads ${SLURM_CPUS_PER_TASK}
read ${MPS_FILE} 
mipopt
write "${RES_FILE}.sol" 
quit
EOF
chmod +x ${CPLEX_COMMAND_SCRIPT}

# Start Cplex with MPI
# On first host, the master is running 
mpirun -np 1 cplex -f ${CPLEX_COMMAND_SCRIPT} -mpi : -np $((SLURM_NTASKS - 1)) cplex -mpi
rm ${CPLEX_COMMAND_SCRIPT}
