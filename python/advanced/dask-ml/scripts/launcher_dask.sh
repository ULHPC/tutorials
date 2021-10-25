#!/bin/bash -l

### Documentation: https://hpc-docs.uni.lu/slurm/launchers/
#SBATCH -p batch           #batch partition
#SBATCH -J DASK_TESTS     #job name
#SBATCH -N 2
#SBATCH --ntasks-per-node  3      # Adapt to maximize node capacity         
#SBATCH -c 1               # 1 core per task
#SBATCH -t 00:30:00        # Job is killed after 10h
###SBATCH --mail-user=firstname.lastname@uni.lu   # Adapt accordingly
###SBATCH --mail-type=BEGIN,FAIL,END

module load lang/Python

export DASK_VENV="./dask_env/bin/activate"


# Source the python env
source ${DASK_VENV}

# Start controller on this first task
dask-scheduler --scheduler-file "./scheduler.json" --interface "ib0" &
sleep 10

#srun: runs ipengine on each available core
srun --cpu-bind=cores dask-worker  --scheduler-file "./scheduler.json" --interface "ib0" &
sleep 25

wait
