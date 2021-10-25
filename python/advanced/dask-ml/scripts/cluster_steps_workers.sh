#!/bin/bash -l

#SBATCH -p batch    
#SBATCH -J DASK_steps_workers    
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4     
#SBATCH --cpus-per-task=1    
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

# Start controller on this first task
dask-scheduler  --scheduler-file "${SCHEDULER_FILE}"  --interface "ib0" &
sleep 10

#srun: runs ipengine on each other available core
srun --cpu-bind=cores dask-worker  \
     --label \
     --interface "ib0" \
     --scheduler-file "${SCHEDULER_FILE}"  &
sleep 25 

python -u $*

