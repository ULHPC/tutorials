#!/bin/bash -l

#SBATCH --job-name="GPU job"
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --time=0-00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --qos=qos-gpu

module restore tf
source ~/venv/tfgpu/bin/activate
srun python imdb-train.py

