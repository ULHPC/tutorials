#!/bin/bash -l

#SBATCH -p batch    
#SBATCH -J DASK_steps_workers    
#SBATCH -N 2
#SBATCH -n 10     
#SBATCH -c 1    
#SBATCH -t 00:30:00    

# Load the python version used to install Dask
module load lang/Python

# Export Environment variables
# Set a environement which depends on which cluster you wish to start the notebook
export VENV="$HOME/.envs/jupyter_dask_${ULHPC_CLUSTER}"

# Replace default jupyter and environement variable by custom ones
# We add to the path the jobid for debugging purpose
export JUPYTER_CONFIG_DIR="$HOME/jupyter/$SLURM_JOBID/"
export JUPYTER_PATH="$VENV/share/jupyter":"$HOME/jupyter_sing/$SLURM_JOBID/jupyter_path"
export JUPYTER_DATA_DIR="$HOME/jupyter/$SLURM_JOBID/jupyter_data"
export JUPYTER_RUNTIME_DIR="$HOME/jupyter/$SLURM_JOBID/jupyter_runtime"

# We create the empty directory
mkdir -p $JUPYTER_CONFIG_DIR

# The Jupyter notebook will run on the first node of the slurm allocation (here only one anyway)
# We retrieve its address
export IP_ADDRESS=$(hostname -I | awk '{print $1}')

# Minimal virtualenv setup
# We create a minimal virtualenv with the necessary packages to start
if [ ! -d "$VENV" ];then
    echo "Building the virtual environment"
    # Create the virtualenv
    python3 -m venv $VENV 
    # Load the virtualenv
    source "$VENV/bin/activate"
    # Upgrade pip 
    python3 -m pip install pip --upgrade
    # Install minimum requirement
    python3 -m pip install jupyter cgroup-utils
    # Setup ipykernel
    # "--sys-prefix" install ipykernel where python is installed
    # here next the python symlink inside the virtualenv
    python3 -m ipykernel install --sys-prefix --name custom_kernel --display-name custom_kernel
fi

export XDG_RUNTIME_DIR=""


# Source the python env 
source "${VENV}/bin/activate"


#create a new ipython profile appended with the job id number
echo "On your laptop: ssh -p 8022 -NL 8889:${IP_ADDRESS}:8889 ${USER}@access-${ULHPC_CLUSTER}.uni.lu " 

# Start jupyter on a single core
srun --exclusive -N 1 -n 1 -c ${SLURM_CPUS_PER_TASK} -w $(hostname) "jupyter notebook --ip ${IP_ADDRESS} --no-browser --port 8889" &


# Dask configuration to store the scheduler file
DASK_CONFIG="${HOME}/.dask"
DASK_JOB_CONFIG="${DASK_CONFIG}/job_${SLURM_JOB_ID}"
mkdir -p ${DASK_JOB_CONFIG}
export SCHEDULER_FILE="${DASK_JOB_CONFIG}/scheduler.json"

# Start scheduler on this first task
srun -w $(hostname) --exclusive -N 1 -n 1 -c ${SLURM_CPUS_PER_TASK} \
     dask-scheduler  --scheduler-file "${SCHEDULER_FILE}"  --interface "ib0" &
sleep 10

# Number of tasks - 1 controller task - 1 jupyter task
export NB_WORKERS=$((${SLURM_NTASKS}-2))

#srun: runs ipengine on each other available core
srun  --exclusive -n ${NB_WORKERS} -c ${SLURM_CPUS_PER_TASK} \
     --cpu-bind=cores dask-worker  \
     --label \
     --interface "ib0" \
     --scheduler-file "${SCHEDULER_FILE}"  &

wait


