#! /bin/bash -l
# Time-stamp: <Mon 2020-12-14 13:05 svarrette>
###############################################################################
# Default launcher for MPI jobs
# Usage:
#       [EXE=/path/to/mpiapp.exe] [sbatch] $0 <mpi-suite> [app].
# By default, MPI programs are expected to be built in APPDIR as '<suite>_<app>'
# to easily get the MPI suite used for the build
################################################################################
#SBATCH -J MPI
###SBATCH --dependency singleton
###SBATCH --mail-type=FAIL     # Mail events (NONE, BEGIN, END, FAIL, ALL)
###SBATCH --mail-user=<email>
#SBATCH --time=0-01:00:00      # 1 hour
#SBATCH --partition=batch
#__________________________
#SBATCH -N 2
#SBATCH --ntasks-per-node 28   # (ideally) as many MPI process as cores available
#SBATCH -c 1                   # multithreading per task : -c --cpus-per-task <n> request
#__________________________
#SBATCH -o logs/%x-%j.out      # log goes into logs/<jobname>-<jobid>.out
mkdir -p logs

##############################
### Guess the run directory
# - either the script directory upon interactive jobs
# - OR the submission directory upon passive/batch jobs
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    [[ "${SCRIPT_DIR}" == *"slurmd"* ]] && TOP_DIR=${SLURM_SUBMIT_DIR} || TOP_DIR=$(realpath -es "${SCRIPT_DIR}")
else
    TOP_DIR="${SCRIPT_DIR}"
fi
CMD_PREFIX=
SUITE='openmpi'
MODULE=mpi/OpenMPI

######################################################
# /!\ ADAPT below variables to match your own settings
APPDIR=${APPDIR:=${HOME}/tutorials/OpenMP-MPI/basics/bin}    # bin directory holding your MPI builds
APP=${APP:=hello_mpi}  # MPI application - OpenMPI/intel/... builds expected to be prefixed by
#                                          openmpi_/intel_/<suit>_
# Eventual options to be passed to the MPI program
OPTS=


################################################################################
print_error_and_exit() { echo "*** ERROR *** $*"; exit 1; }
usage() {
    cat <<EOF
NAME
  $(basename $0): Generic MPI launcher
    Default APPDIR: ${APPDIR}
    Default APP: ${APP}
  Take the good habit to prefix the binary to execute with MPI suit used for
  the build. Here the default MPI application run would be
        EXE=${APPDIR}/openmpi_${APP}
  which will be run as     srun -n \$SLURM_NTASKS [...]

USAGE
  [sbatch] $0 [-n] {intel | openmpi | mvapich2} [app]
  EXE=/path/to/mpiapp.exe [sbatch] $0 [-n] {intel | openmpi | mvapich2}

OPTIONS:
  -n --dry-run: Dry run mode

Example:
  [sbatch] $0                          # run OpenMPI build    openmpi_hello_mpi
  [sbatch] $0 intel                    # run Intel MPI build  intel_hello_mpi
  [sbatch] $0 openmpi matrix_mult_mpi  # run OpenMPI build    openmpi_matrix_mult_mpi
  EXE=$HOME/bin/xhpl [sbatch] $0 intel # run intel build  ~/bin/xhpl
EOF
}

################################################################################
while [ $# -ge 1 ]; do
    case $1 in
        -h | --help) usage; exit 0;;
        -n | --noop | --dry-run) CMD_PREFIX=echo;;
        intel*   | --intel*)   SUITE='intel';    MODULE=toolchain/intel;;
        openmpi* | --openmpi*) SUITE='openmpi';  MODULE=mpi/OpenMPI;;
        mvapich* | --mvapich*) SUITE='mvapich2'; MODULE=mpi/MVAPICH2;;
        *) APP=$1; shift; OPTS=$*; break;;
    esac
    shift
done
################################################################################
# Set default MPI executable
[ -z "${EXE}" ]    && EXE=${APPDIR}/${SUITE}_${APP} || APP=$(basename ${EXE})
[ ! -x "${EXE}" ]  && print_error_and_exit "Unable to find the executable ${EXE}"

cat <<EOF
# ==============================================================
# => MPI run of '${APP}' with the ${SUITE} MPI suite
# ==============================================================
EOF

module purge || print_error_and_exit "Unable to find the module command"
module load ${MODULE}
module list

start=$(date +%s)
echo "### Starting timestamp (s): ${start}"

${CMD_PREFIX} srun -n ${SLURM_NTASKS:-1} ${EXE} ${OPTS}

end=$(date +%s)
cat <<EOF
### Ending timestamp (s): ${end}"
# Elapsed time (s): $(($end-$start))
EOF
