#! /bin/bash -l
# Time-stamp: <Sat 2018-06-09 19:33 svarrette>
######## OAR directives ########
#OAR -n OpenMP
#OAR -l nodes=1/core=4,walltime=0:05:00
#OAR -O OpenMP-%jobid%.log
#OAR -E OpenMP-%jobid%.log
#
####### Slurm directives #######
#SBATCH -J OpenMP
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --time=0-00:05:00
#SBATCH -p batch
#SBATCH --qos=qos-batch

################################################################################
# OpenMP Setup
if [ -n "$OAR_NODEFILE" ]; then
    export OMP_NUM_THREADS=$(cat $OAR_NODEFILE| wc -l)
elif [ -n "SLURM_CPUS_PER_TASKS" ]; then
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
fi
# Use the UL HPC modules
if [ -f  /etc/profile ]; then
   .  /etc/profile
fi
################################################################################
# Directory holding your built applications
# /!\ ADAPT to match your own settings
APPDIR="$HOME/tutorials/OpenMP-MPI/basics/bin"
APP=hello_openmp

echo "=> Executing OpenMP program '${APP}' with the 'foss' toolchain"
module purge
module load toolchain/foss
${APPDIR}/${APP}


echo "=> Executing OpenMP program ${APP} with the 'intel' toolchain"
module purge
module load toolchain/intel
${APPDIR}/intel_${APP}
