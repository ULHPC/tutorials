#!/bin/bash -l
#SBATCH -J Singularity_Jupyter_parallel_cuda
#SBATCH -N 1 # Nodes
#SBATCH -n 1 # Tasks
#SBATCH -c 1 # Cores assigned to each tasks
#SBATCH --time=0-01:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --qos=normal
#SBATCH --mail-user=<firstname>.<lastname>@uni.lu
#SBATCH --mail-type=BEGIN,END



module load tools/Singularity

export VENV="$HOME/.envs/venv_parallel_cuda"
export JUPYTER_CONFIG_DIR="$HOME/jupyter_sing/$SLURM_JOBID/"
export JUPYTER_PATH="$VENV/share/jupyter":"$HOME/jupyter_sing/$SLURM_JOBID/jupyter_path"
export JUPYTER_DATA_DIR="$HOME/jupyter_sing/$SLURM_JOBID/jupyter_data"
export JUPYTER_RUNTIME_DIR="$HOME/jupyter_sing/$SLURM_JOBID/jupyter_runtime"
export IPYTHONDIR="$HOME/ipython_sing/$SLURM_JOBID"

mkdir -p $JUPYTER_CONFIG_DIR
mkdir -p $IPYTHONDIR

export IP_ADDRESS=$(hostname -I | awk '{print $1}')


echo "On your laptop: ssh -p 8022 -NL 8889:${IP_ADDRESS}:8889 ${USER}@access-${ULHPC_CLUSTER}.uni.lu " 


singularity instance start --nv jupyter_parallel_cuda.sif jupyter

if [ ! -d "$VENV" ];then
    # For some reasons, there is an issue with venv -- using virtualenv instead
    singularity exec --nv instance://jupyter python3 -m virtualenv $VENV --system-site-packages
    singularity run --nv instance://jupyter $VENV "python3 -m pip install --upgrade pip" 
    # singularity run --nv instance://jupyter $VENV "python3 -m pip install <your_packages>"
    singularity run --nv instance://jupyter $VENV "python3 -m ipykernel install --sys-prefix --name HPC_SCHOOL_ENV_IPYPARALLEL_CUDA --display-name HPC_SCHOOL_ENV_IPYPARALLEL_CUDA"

fi

#create a new ipython profile appended with the job id number
profile=job_${SLURM_JOB_ID}
singularity run --nv instance://jupyter $VENV "ipython profile create --parallel ${profile}"

# Enable IPython clusters tab in Jupyter notebook
singularity run --nv instance://jupyter $VENV "jupyter nbextension enable --py ipyparallel"

## Start Controller and Engines
#
singularity run --nv instance://jupyter $VENV "ipcontroller --ip="*" --profile=${profile}" &
sleep 10

##srun: runs ipengine on each available core
srun singularity run --nv jupyter_parallel.sif $VENV "ipengine --profile=${profile} --location=$(hostname)" &
sleep 25

export XDG_RUNTIME_DIR=""

singularity run --nv instance://jupyter $VENV "jupyter notebook --ip ${IP_ADDRESS} --no-browser --port 8889" &
pid=$!
sleep 5s
singularity run --nv instance://jupyter $VENV "jupyter notebook list"
singularity run --nv instance://jupyter $VENV "jupyter --paths"
singularity run --nv instance://jupyter $VENV "jupyter kernelspec list"

wait $pid
echo "Stopping instance"
singularity instance stop jupyter



