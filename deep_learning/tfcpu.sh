#!/bin/bash -l

#SBATCH --job-name="CPU job"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=0-00:10:00
#SBATCH --partition=batch
#SBATCH --qos=qos-batch

module restore tf
source ~/venv/tfcpu/bin/activate
squeue -l -j $SLURM_JOB_ID
srun python imdb-train.py

