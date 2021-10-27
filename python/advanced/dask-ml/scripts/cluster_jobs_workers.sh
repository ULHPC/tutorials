#!/bin/bash -l

#SBATCH -p batch
#SBATCH -J DASK_jobs_workers
#SBATCH --nodes=1
#SBATCH --ntasks=1
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

export NB_WORKERS=10

# Source the python env
source "${DASK_VENV}/bin/activate"

python -u $*


