#!/bin/bash -l
#SBATCH -J cifar-cnn
#SBATCH -o logs/%x-%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 1:0:0
#SBATCH -p batch
##SBATCH --reservation=hpcschool

. scripts/setup.sh
config=configs/cifar10_cnn.yaml
srun python train.py $config
