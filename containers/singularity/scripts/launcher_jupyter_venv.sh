#!/bin/bash -l
#SBATCH -J Singularity_Jupyter_venv
#SBATCH -N 1 # Nodes
#SBATCH -n 1 # Tasks
#SBATCH -c 2 # Cores assigned to each tasks
#SBATCH --time=0-01:00:00
#SBATCH -p batch
#SBATCH --qos=normal
#SBATCH --mail-user=<firstname>.<lastname>@uni.lu
#SBATCH --mail-type=BEGIN,END



module load tools/Singularity
export VENV="$HOME/.envs/venv_${ULHPC_CLUSTER}"
export JUPYTER_CONFIG_DIR="$HOME/jupyter_sing/$SLURM_JOBID/"
export JUPYTER_PATH="$VENV/share/jupyter":"$HOME/jupyter_sing/$SLURM_JOBID/jupyter_path"
export JUPYTER_DATA_DIR="$HOME/jupyter_sing/$SLURM_JOBID/jupyter_data"
export JUPYTER_RUNTIME_DIR="$HOME/jupyter_sing/$SLURM_JOBID/jupyter_runtime"
export IPYTHONDIR="$HOME/ipython_sing/$SLURM_JOBID"

mkdir -p $JUPYTER_CONFIG_DIR
mkdir -p $IPYTHONDIR

export IP_ADDRESS=$(hostname -I | awk '{print $1}')


echo "On your laptop: ssh -p 8022 -NL 8889:${IP_ADDRESS}:8889 ${USER}@access-${ULHPC_CLUSTER}.uni.lu " 

singularity instance start jupyter_venv.sif jupyter

if [ ! -d "$VENV" ];then
    singularity exec instance://jupyter python3 -m venv $VENV --system-site-packages
    # singularity run instance://jupyter $VENV "python3 -m pip install <your_packages>"
    singularity run instance://jupyter $VENV "python3 -m ipykernel install --sys-prefix --name HPC_SCHOOL_ENV --display-name HPC_SCHOOL_ENV"

fi



singularity run instance://jupyter  $VENV "jupyter notebook --ip ${IP_ADDRESS} --no-browser --port 8889" &
pid=$!
sleep 5s
singularity run instance://jupyter $VENV "jupyter notebook list"
singularity run instance://jupyter $VENV "jupyter --paths"
singularity run instance://jupyter $VENV "jupyter kernelspec list"
wait $pid
echo "Stopping instance"
singularity instance stop jupyter



