#!/bin/bash -l
# Time-stamp: <Sun 2017-06-11 22:13 svarrette>
##################################################################

##########################
#                        #
#  The SLURM directives  #
#                        #
##########################
#
#          Set number of resources
#
#SBATCH -N 1
# Exclusive mode is recommended for all spark jobs
#SBATCH --exclusive

#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 28

###SBATCH -n 1
### -c, --cpus-per-task=<ncpus>
###     (multithreading) Request that ncpus be allocated per process
###SBATCH -c 28
#SBATCH --time=0-01:00:00   # 1 hour
#
#          Set the name of the job
#SBATCH -J SparkMaster
#          Passive jobs specifications
#SBATCH --partition=batch
#SBATCH --qos qos-batch

### General SLURM Parameters
echo "SLURM_JOBID  = ${SLURM_JOBID}"
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST}"
echo "SLURM_NNODES = ${SLURM_NNODES}"
echo "SLURM_NTASK  = ${SLURM_NNODES}"
echo "SLURMTMPDIR  = ${SLURMTMPDIR}"
echo "Submission directory = ${SLURM_SUBMIT_DIR}"

### Guess the run directory
# - either the script directory upon interactive jobs
# - OR the submission directory upon passive/batch jobs
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    [[ "${SCRIPTDIR}" == *"slurmd"* ]] && RUNDIR=${SLURM_SUBMIT_DIR} || RUNDIR=${SCRIPTDIR}
else
    RUNDIR=${SCRIPTDIR}
fi

### Prepare log file

### Toolbox function
print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }

############################################
################# Let's go #################
############################################

# Use the RESIF build modules
if [ -f  /etc/profile ]; then
   .  /etc/profile
fi

# Load the {intel | foss} toolchain and whatever module(s) you need
module purge
module use $HOME/.local/easybuild/modules/all
module load devel/Spark

export SPARK_HOME=$EBROOTSPARK

# sbin/start-master.sh - Starts a master instance on the machine the script is executed on.
$SPARK_HOME/sbin/start-all.sh

export MASTER=spark://$HOSTNAME:7077

echo
echo "========= Spark Master ========"
echo $MASTER
echo "==============================="

spark-submit --master=$MASTER $SPARK_HOME/examples/src/main/python/pi.py 50



#tail -f /dev/null #wait forever

# sbin/stop-master.sh - Stops the master that was started via the bin/start-master.sh script.
$SPARK_HOME/sbin/stop-all.sh


