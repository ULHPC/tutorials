#!/bin/bash -l
#SBATCH --job-name=example3
#SBATCH --output=example3.out
#
#SBATCH --ntasks=1
#SBATCH --time=0-00:10:00
#SBATCH -p batch
#SBATCH --qos=normal

module load vis/gnuplot

python example1.py
# Activate virtualenv that contains numpy
source numpy16/bin/activate
python example3.py
deactivate
gnuplot gnuplot/time_vs_array_size.gpi
