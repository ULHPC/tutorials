#!/bin/bash -l
#SBATCH -J imagenet-resnet
#SBATCH -o logs/%x-%j.out
#SBATCH -N 1
#SBATCH -n 28
#SBATCH -t 1:0:0
#SBATCH -p batch
##SBATCH --reservation=hpcschool

. scripts/setup.sh
config=configs/imagenet_resnet.yaml
srun -l python train.py $config --distributed
