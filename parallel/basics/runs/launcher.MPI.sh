#! /bin/bash -l
# Time-stamp: <Sun 2018-06-10 14:33 svarrette>
######## OAR directives ########
#OAR -n MPI
#OAR -l nodes=2/core=3,walltime=0:05:00
#OAR -O MPI-%jobid%.log
#OAR -E MPI-%jobid%.log
#
####### Slurm directives #######
#SBATCH -J MPI
#SBATCH -N 2
#SBATCH --ntasks-per-node=3
#SBATCH -c 1
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
  $0 -- MPI launcher example
USAGE
  $0 {intel | openmpi | mvapich2} [app]

Example:
  $0                          run OpenMPI on hello_mpi
  $0 intel                    run Intel MPI on hello_mpi
  $0 openmpi matrix_mult_mpi  run OpenMPI on matrix_mult_mpi
EOF
}

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
[ -n "$2" ] && APP=${SUITE}_$2 || APP=${SUITE}_hello_mpi
# Eventual options of the MPI program
OPTS=

EXE="${APPDIR}/${APP}"
[ ! -x "${EXE}" ]  && print_error_and_exit "Unable to find the generated executable ${EXE}"


echo "# =============================================================="
echo "# => MPI run of '${APP}' with the ${SUITE} MPI suite            "
echo "# =============================================================="

module purge || print_error_and_exit "Unable to find the module command"
module load ${MODULE}
module list

# The command to run
case ${SUITE} in
    openmpi)  CMD="mpirun -hostfile \$OAR_NODEFILE -x PATH -x LD_LIBRARY_PATH ${EXE} ${OPTS}";;
    mvapich*) CMD="mpirun -launcher ssh -launcher-exec /usr/bin/oarsh -f \$OAR_NODEFILE ${EXE} ${OPTS}";;
    intel)    CMD="mpirun -hostfile \$OAR_NODEFILE ${EXE} ${OPTS}";;
    *)  print_error_and_exit "Unsupported MPI suite";;
esac
# Easier on slurm...
if [ -n "${SLURM_NTASKS}" ]; then
    CMD="srun -n \${SLURM_NTASKS} ${EXE} ${OPTS}"
fi

echo "=> running command: ${CMD}"
eval ${CMD}
