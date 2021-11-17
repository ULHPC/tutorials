#! /bin/bash -l
# Time-stamp: <Wed 2021-11-17 16:46 svarrette>
################################################################################
# Default launcher for OpenMP jobs
# Usage:
#       [EXE=/path/to/multithreadedapp.exe] [sbatch] $0 <mpi-suite> [app].
# By default, OpenMP programs are expected to be built in APPDIR.
# Intel builds are expected to be named 'intel_<app>' to easily get the toolchain
# used for the build
#################################################################################
#SBATCH -J OpenMP
###SBATCH --dependency singleton
###SBATCH --mail-type=FAIL     # Mail events (NONE, BEGIN, END, FAIL, ALL)
###SBATCH --mail-user=<email>
#SBATCH --time=0-01:00:00      # 1 hour
#SBATCH --partition=batch
#__________________________
#SBATCH -N 1
#SBATCH --ntasks-per-node 1    #
#SBATCH -c 28                  # multithreading per task : -c --cpus-per-task <n> request
#__________________________    #      (ideally) as many OpenMP threads as cores available
#                              #  In praticular, set to 128 on Aion
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
MODULE=toolchain/foss

######################################################
# /!\ ADAPT below variables to match your own settings
APPDIR=${APPDIR:=${HOME}/tutorials/OpenMP-MPI/basics/bin}    # bin directory holding your OpenMP builds
APP=${APP:=${ULHPC_CLUSTER}_hello_openmp}     # OpenMP application - intel builds expected to be prefixed by intel_<APP>
# Eventual options to be passed to the MPI program
OPTS=


#
### Usage:
# $0 intel  [app]
# $0 foss   [app]

################################################################################
print_error_and_exit() { echo "*** ERROR *** $*"; exit 1; }
usage() {
    cat <<EOF
NAME
  $(basename $0): Generic OpenMP launcher
    Default APPDIR: ${APPDIR}
    Default APP: ${APP}
  Take the good habit to prefix the intel binaries (as foss toolchain is assumed by default)
  with 'intel_'

USAGE
  [sbatch] $0 [-n] {intel | foss } [app]
  EXE=/path/to/multithreadedapp.exe [sbatch] $0 [-n] {intel | foss }

OPTIONS:
  -n --dry-run: Dry run mode

Example:
  [sbatch] $0                          # run FOSS  build   <cluster>_hello_openmp
  [sbatch] $0 intel                    # run intel build   intel_<cluster>_hello_openmp
  [sbatch] $0 foss matrix_mult_openmp  # run FOSS  build   matrix_mult_openmp
  EXE=$HOME/bin/datarace [sbatch] $0 intel # run intel build  ~/bin/datarace
EOF
}

################################################################################
while [ $# -ge 1 ]; do
    case $1 in
        -h | --help) usage; exit 0;;
        -n | --noop | --dry-run) CMD_PREFIX=echo;;
        intel* | --intel*) MODULE=toolchain/intel;;
        foss*  | --foss*)  MODULE=toolchain/foss;;
        *) APP=$1; shift; OPTS=$*; break;;
    esac
    shift
done
################################################################################
# OpenMP Setup
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

[[ "${MODULE}" == *"intel"* ]] && APP=intel_${APP}
[ -z "${EXE}" ]   && EXE=${APPDIR}/${APP} || APP=$(basename ${EXE})
[ ! -x "${EXE}" ] && print_error_and_exit "Unable to find the executable ${EXE}"

cat <<EOF
# ==============================================================
# => OpenMP run of '${APP}' with ${MODULE}
#    OMP_NUM_THREADS=${OMP_NUM_THREADS}
# ==============================================================
EOF

module purge || print_error_and_exit "Unable to find the module command"
module load ${MODULE}
module list

start=$(date +%s)
echo "### Starting timestamp (s): ${start}"

${CMD_PREFIX} srun ${EXE} ${OPTS}

end=$(date +%s)
cat <<EOF
### Ending timestamp (s): ${end}"
# Elapsed time (s): $(($end-$start))
EOF
