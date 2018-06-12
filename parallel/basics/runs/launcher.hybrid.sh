#! /bin/bash -l
# Time-stamp: <Sun 2018-06-10 22:01 svarrette>
######## OAR directives ########
#OAR -n Hybrid
#OAR -l nodes=2/core=4,walltime=0:05:00
#OAR -O Hybrid-%jobid%.log
#OAR -E Hybrid-%jobid%.log
#
####### Slurm directives #######
#SBATCH -J Hybrid
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --time=0-00:05:00
#SBATCH -p batch
#SBATCH --qos=qos-batch

# Usage:
# $0 intel     [app]
# $0 openmpi   [app]
# $0 mvapich2  [app]

################################################################################
print_error_and_exit() { echo "*** ERROR *** $*"; exit 1; }
usage() {
    cat <<EOF
NAME
  $0 -- Hybrid OpenMP+MPI launcher example
USAGE
  $0 {intel | openmpi | mvapich2} [app]

Example:
  $0 [openmpi]      run hybrid OpenMP+OpenMPI  on hello_hybrid
  $0 intel          run hybrid OpenMP+IntelMPI on hello_hybrid
  $0 mvapich2       run hybrid OpenMP+MVAPICH2 on hello_hybrid
EOF
}

################################################################################
# OpenMP+MPI Setup
if [ -n "$OAR_NODEFILE" ]; then
    NTASKS=$(cat $OAR_NODEFILE | wc -l)
    NNODES=$(cat $OAR_NODEFILE | sort -u | wc -l)
    NCORES_PER_NODE=$(echo "${NTASKS}/${NNODES}" | bc)
    export OMP_NUM_THREADS=$(cat $OAR_NODEFILE | uniq -c | head -n1 | awk '{print $1}')
    NPERNODE=$(echo "$NCORES_PER_NODE/$OMP_NUM_THREADS" | bc)
    NP=$(echo "$NTASKS/$OMP_NUM_THREADS" | bc)
    MACHINEFILE=hostfile_${OAR_JOBID}.txt;
    # Unique list of hostname for the machine file
    cat $OAR_NODEFILE | uniq > ${MACHINEFILE};
elif [ -n "SLURM_CPUS_PER_TASKS" ]; then
    NCORES_PER_NODE=$(echo "${SLURM_NTASKS}/${$SLURM_NNODES}" | bc)
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
    NPERNODE=$(echo "$NCORES_PER_NODE/$OMP_NUM_THREADS" | bc)
    NP=$SLURM_NTASKS
    MACHINEFILE=hostfile_${SLURM_JOBID}.txt;
    # Unique list of hostname for the machine file
    srun hostname | sort -n | uniq > ${MACHINEFILE};
fi
# Use the UL HPC modules
if [ -f  /etc/profile ]; then
   .  /etc/profile
fi

################################################################################
case $1 in
    -h       | --help)     usage; exit 0;;
    intel*   | --intel*)   SUITE='intel';    MODULE=toolchain/intel;;
    mvapich* | --mvapich*) SUITE='mvapich2'; MODULE=mpi/MVAPICH2;;
    *)                     SUITE='openmpi';  MODULE=mpi/OpenMPI;;
esac

################################################################################
################################################################################
# Directory holding your built applications
# /!\ ADAPT to match your own settings
APPDIR="$HOME/tutorials/OpenMP-MPI/basics/bin"
[ -n "$2" ] && APP=${SUITE}_$2 || APP=${SUITE}_hello_hybrid
# Eventual options of the MPI program
OPTS=

EXE="${APPDIR}/${APP}"
[ ! -x "${EXE}" ]  && print_error_and_exit "Unable to find the generated executable ${EXE}"


echo "# =============================================================="
echo "# => OpenMP+MPI run of '${APP}' with the ${SUITE} MPI suite"
echo "#    OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "#    ${NP} MPI process(es)"
echo "# =============================================================="

module purge || print_error_and_exit "Unable to find the module command"
module load ${MODULE}
module list

# The command to run
case ${SUITE} in
    openmpi)
        CMD="mpirun -npernode ${NPERNODE:=1} -np ${NP} -hostfile \$OAR_NODEFILE -x OMP_NUM_THREADS -x PATH -x LD_LIBRARY_PATH ${EXE} ${OPTS}";;
    mvapich*)
        export MV2_ENABLE_AFFINITY=0;
        CMD="mpirun -ppn ${NPERNODE:=1} -np ${NP} -genv OMP_NUM_THREADS=${OMP_NUM_THREADS} -launcher ssh -launcher-exec /usr/bin/oarsh -f $MACHINEFILE ${EXE} ${OPTS}";;
    intel)
        CMD="mpirun -perhost ${NPERNODE:=1} -np ${NP} -genv OMP_NUM_THREADS=${OMP_NUM_THREADS} -genv I_MPI_PIN_DOMAIN=omp -hostfile \$OAR_NODEFILE ${EXE} ${OPTS}";;
    *)  print_error_and_exit "Unsupported MPI suite";;
esac
# Way easier on slurm...
if [ -n "${SLURM_NTASKS}" ]; then
    CMD="srun -n \${SLURM_NTASKS} ${EXE} ${OPTS}"
fi

echo "=> running command: ${CMD}"
eval ${CMD}

rm -f ${MACHINEFILE}
