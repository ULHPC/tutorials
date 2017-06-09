#!/bin/bash -l
#SBATCH --job-name=scoop
#SBATCH --output=scoop.log
#
#SBATCH --ntasks=1
#SBATCH --time=0-00:05:00
#SBATCH -p batch
#SBATCH --qos=qos-batch

python setup.py build_ext --inplace
# Cython code execution
#time python -c "import pi_calc; pi_calc.run()"
# Cython numpy code execution
time python -c "import pi_calc_numpy; pi_calc_numpy.run()"
# Normal Python execution
#time python pi_calc.py
# Normal Numpy Python execution
#time python pi_calc_numpy.py
