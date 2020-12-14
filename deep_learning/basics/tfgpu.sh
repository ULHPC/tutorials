#!/bin/bash -l

#SBATCH --job-name="GPU imdb"
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --time=0-00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

module restore tf
source ~/venv/tfgpu/bin/activate
srun python imdb-train.py

