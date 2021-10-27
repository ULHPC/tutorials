#!/bin/bash -l

#SBATCH -p batch    
#SBATCH -J DASK_steps_workers    
#SBATCH -N 2
#SBATCH -n 10     
#SBATCH -c 1    
#SBATCH -t 00:30:00    

# Load the python version used to install Dask
module load lang/Python

# Make sure that you have an virtualenv dask_env installed
export DASK_VENV="$1" 
shift
if [ ! -d "${DASK_VENV}" ] || [ ! -f "${DASK_VENV}/bin/activate" ]; then
        
        echo "Error with virtualenv" && exit 1

    fi

# Source the python env 
source "${DASK_VENV}/bin/activate"

# Dask configuration to store the scheduler file
DASK_CONFIG="${HOME}/.dask"
DASK_JOB_CONFIG="${DASK_CONFIG}/job_${SLURM_JOB_ID}"
mkdir -p ${DASK_JOB_CONFIG}
export SCHEDULER_FILE="${DASK_JOB_CONFIG}/scheduler.json"


# Number of tasks - 1 controller task - 1 python task
export NB_WORKERS=$((${SLURM_NTASKS}-2))


LOG_DIR="$(pwd)/logs/job_${SLURM_JOBID}"
mkdir -p ${LOG_DIR}

# Start scheduler on this first task
srun -w $(hostname) --output=${LOG_DIR}/scheduler-%j-workers.out  --exclusive -N 1 -n 1 -c ${SLURM_CPUS_PER_TASK} \
     dask-scheduler  --scheduler-file "${SCHEDULER_FILE}"  --interface "ib0" &
sleep 10

#srun: runs ipengine on each other available core
srun --output=${LOG_DIR}/ipengine-%j-workers.out \
     --exclusive -n ${NB_WORKERS} -c ${SLURM_CPUS_PER_TASK} \
     --cpu-bind=cores dask-worker  \
     --label \
     --interface "ib0" \
     --scheduler-file "${SCHEDULER_FILE}"  &

sleep 25 

srun --output=${LOG_DIR}/code-%j-execution.out  --exclusive -N 1 -n 1 -c ${SLURM_CPUS_PER_TASK} python -u $*


