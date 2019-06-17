#!/bin/bash -l
#SBATCH -J Multi-threaded_cplex
#SBATCH --tasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=0-01:00:00
#SBATCH -p batch
#SBATCH --qos=qos-batch

# Load cplex 
module load math/CPLEX/12.8-foss-2018a

# Some variable
MPS_FILE=$1
RES_FILE=$2
CPLEX_COMMAND_SCRIPT="command_job${SLURM_JOBID}.lst"



# Create cplex command script
cat << EOF > ${CPLEX_COMMAND_SCRIPT}
set threads ${SLURM_CPUS_PER_TASK}
read ${MPS_FILE} 
mipopt
write "${RES_FILE}.sol" 
quit
EOF
chmod +x ${CPLEX_COMMAND_SCRIPT}

# Cplex will access use the required number of thread
cplex -f ${CPLEX_COMMAND_SCRIPT}
rm ${CPLEX_COMMAND_SCRIPT}
