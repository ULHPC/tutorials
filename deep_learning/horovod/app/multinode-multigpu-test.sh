#!/bin/sh –l
#SBATCH -c 2              # 2 CPU-core for each process
#SBATCH -N 2              # 2 nodes
#SBATCH -p gpu
#SBATCH --gpus-per-node 4 # Each process is associated to each GPU
#SBATCH -t 30
#SBATCH --export=ALL

mpirun -n 8 python test_horovod.py

