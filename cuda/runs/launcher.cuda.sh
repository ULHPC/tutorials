#!/bin/bash -l
#SBATCH --job-name="GPU build"
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --time=0-00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

if [ -z "$1" ]
then
	echo "Missing required source (.cu), and optional execution arguments."
	exit
fi

src=${1}
exe=$(basename ${1/cu/out})
ptx=$(basename ${1/cu/ptx})
prf=$(basename ${1/cu/prof})
shift
args=$*

module restore cuda

srun nvcc -arch=compute_70 -o ./$exe $src
srun nvcc -ptx -arch=compute_70 -o ./$ptx $src
srun ./$exe $args
srun nvprof --log-file ./$prf ./$exe $args
echo "file: $prf"
cat ./$prf
