#!/bin/bash -l
#SBATCH -J Singularity_Jupyter_parallel
#SBATCH -N 2 # Nodes
#SBATCH -n 10 # Tasks
#SBATCH -c 2 # Cores assigned to each tasks
#SBATCH --time=0-01:00:00
#SBATCH -p batch
#SBATCH --qos=normal
#SBATCH --mail-user=<firstname>.<lastname>@uni.lu
#SBATCH --mail-type=BEGIN,END



module load tools/Singularity

export VENV="$HOME/.envs/venv_parallel_${ULHPC_CLUSTER}"
export JUPYTER_CONFIG_DIR="$HOME/jupyter_sing/$SLURM_JOBID/"
export JUPYTER_PATH="$VENV/share/jupyter":"$HOME/jupyter_sing/$SLURM_JOBID/jupyter_path"
export JUPYTER_DATA_DIR="$HOME/jupyter_sing/$SLURM_JOBID/jupyter_data"
export JUPYTER_RUNTIME_DIR="$HOME/jupyter_sing/$SLURM_JOBID/jupyter_runtime"
export IPYTHONDIR="$HOME/ipython_sing/$SLURM_JOBID"

mkdir -p $JUPYTER_CONFIG_DIR
mkdir -p $IPYTHONDIR


export IP_ADDRESS=$(hostname -I | awk '{print $1}')
export XDG_RUNTIME_DIR=""

#create a new ipython profile appended with the job id number
profile=job_${SLURM_JOB_ID}

JUPYTER_SRUN="srun -w $(hostname) --exclusive -N 1 -n 1 -c 1 "
IPCONTROLLER_SRUN="srun -w $(hostname) --exclusive -N 1 -n 1 -c 1 "
IPENGINES_SRUN="srun --exclusive -n $((${SLURM_NTASKS}-2)) -c ${SLURM_CPUS_PER_TASK} "


echo "On your laptop: ssh -p 8022 -NL 8889:${IP_ADDRESS}:8889 ${USER}@access-${ULHPC_CLUSTER}.uni.lu " 

if [ ! -d "$VENV" ];then
    ${JUPYTER_SRUN} -J "JUP: Create venv" singularity exec jupyter_parallel.sif python3 -m venv $VENV --system-site-packages
    # singularity run jupyter_parallel.sif $VENV "python3 -m pip install <your_packages>"
    ${JUPYTER_SRUN} -J "JUP: Install ipykernel" singularity run jupyter_parallel.sif $VENV "python3 -m ipykernel install --sys-prefix --name HPC_SCHOOL_ENV_IPYPARALLEL --display-name HPC_SCHOOL_ENV_IPYPARALLEL"

fi

${JUPYTER_SRUN} -J "JUP: Create profile" singularity run jupyter_parallel.sif $VENV "ipython profile create --parallel ${profile}"

# Enable IPython clusters tab in Jupyter notebook
${JUPYTER_SRUN} -J "JUP: add ipy ext" singularity run jupyter_parallel.sif $VENV "jupyter nbextension enable --py ipyparallel"

${JUPYTER_SRUN} -J "JUP: Start jupyter notebook" singularity run jupyter_parallel.sif $VENV "jupyter notebook --ip ${IP_ADDRESS} --no-browser --port 8889" &

sleep 5s
${JUPYTER_SRUN}  -J "JUP: Get notebook list" singularity run jupyter_parallel.sif $VENV "jupyter notebook list"
${JUPYTER_SRUN} -J "JUP: Get jupyter paths info" singularity run jupyter_parallel.sif $VENV "jupyter --paths"
${JUPYTER_SRUN} -J "JUP: Get jupyter kernels" singularity run jupyter_parallel.sif $VENV "jupyter kernelspec list"



## Ipyparallel for ditributed notebook

## Start Controller and Engines
${IPCONTROLLER_SRUN} -J "IPCONTROLLER" singularity run jupyter_parallel.sif $VENV "ipcontroller --ip="*" --profile=${profile}" &
sleep 10

##srun: runs ipengine on each available core
${IPENGINES_SRUN} -J "IPENGINES" singularity run jupyter_parallel.sif $VENV "ipengine --profile=${profile} --location=$(hostname)" &
sleep 25

wait






