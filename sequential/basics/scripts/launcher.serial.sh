#! /bin/bash -l
# Time-stamp: <Fri 2020-12-11 00:22 svarrette>
############################################################################
# Default launcher for serial (one core) tasks
############################################################################
###SBATCH -J Serial-jobname
###SBATCH --dependency singleton
###SBATCH --mail-type=FAIL     # Mail events (NONE, BEGIN, END, FAIL, ALL)
###SBATCH --mail-user=<email>
#SBATCH --time=0-01:00:00      # 1 hour
#SBATCH --partition=batch
#__________________________
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
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

# /!\ ADAPT TASK variable accordingly
# Absolute path to the (serial) task to be executed i.e. your favorite
# Java/C/C++/Ruby/Perl/Python/R/whatever program to be run
TASK=${TASK:=${HOME}/bin/app.exe}

################################################################################
print_error_and_exit() { echo "*** ERROR *** $*"; exit 1; }
usage() {
    cat <<EOF
NAME
  $(basename $0): Generic launcher for the serial application
     Default TASK: ${TASK}
USAGE
  [sbatch] $0 [-n]
  TASK=/path/to/app.exe [sbatch] $0 [-n]
OPTIONS:
  -n --dry-run: Dry run mode
EOF
}
print_debug_info(){
cat <<EOF
 TOP_DIR    = ${TOP_DIR}
 TASK       = ${TASK}
EOF
[ -n "${SLURM_JOBID}" ] && echo "$(scontrol show job ${SLURM_JOBID})"
}
################################################################################
# Check for options
while [ $# -ge 1 ]; do
    case $1 in
        -h | --help) usage; exit 0;;
        -d | --debug) print_debug_info;;
        -n | --noop | --dry-run) CMD_PREFIX=echo;;
        *) OPTS=$*; break;;
    esac
    shift
done
[ ! -x "${TASK}" ] && print_error_and_exit "Unable to find TASK=${TASK}"
module purge || print_error_and_exit "Unable to find the 'module' command"
### module load [...]
# module load lang/Python
# source ~/venv/<name>/bin/activate

start=$(date +%s)
echo "### Starting timestamp (s): ${start}"

${CMD_PREFIX} ${TASK} ${OPTS} ${SLURM_ARRAY_TASK_ID}

end=$(date +%s)
cat <<EOF
### Ending timestamp (s): ${end}"
# Elapsed time (s): $(($end-$start))
EOF
