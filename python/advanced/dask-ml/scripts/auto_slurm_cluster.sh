#!/bin/bash -l

#SBATCH -p batch
#SBATCH -J DASK_main_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:30:00

# Load the python version used to install Dask
module load lang/Python

# Make sure that you have an virtualenv dask_env installed
export DASK_VENV="./dask_env/bin/activate"

# Source the python env
source ${DASK_VENV}

python -u $*
