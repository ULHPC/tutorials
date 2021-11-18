#!/bin/bash -l
#SBATCH --job-name=example1
#SBATCH --output=example1.out
#
#SBATCH --ntasks=1
#SBATCH --time=0-00:10:00
#SBATCH -p batch
#SBATCH --qos=normal

module load vis/gnuplot

python example1.py
gnuplot gnuplot/time_vs_array_size.gpi
