#!/bin/bash -l

#SBATCH --job-name="CPU imdb"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:10:00
#SBATCH --partition=batch
#SBATCH --qos=normal

module restore tf
source ~/venv/tfcpu/bin/activate
srun python imdb-train.py
