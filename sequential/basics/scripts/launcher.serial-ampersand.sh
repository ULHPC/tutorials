#! /bin/bash -l
# Time-stamp: <Fri 2020-12-11 00:43 svarrette>
############################################################################
# (Not Recommended) Sample launcher for aggregating serial (one core) tasks
# within one node using the Bash & (ampersand), a builtin control operator
# used to fork processes, and the wait command.
############################################################################
#SBATCH -J StressMe-ampersand
#SBATCH --time=0-01:00:00      # 1 hour
#SBATCH --partition=batch
#__________________________
#SBATCH -N 1
#SBATCH --ntasks-per-node 28   # <-- match number of cores per node
#SBATCH -c 1                   # multithreading per task : -c --cpus-per-task <n> request
#__________________________
#SBATCH -o logs/%x-%j.out      # log goes into logs/<jobname>-<jobid>.out
mkdir -p logs

CMD_PREFIX=
# the --exclusive to srun makes srun use distinct CPUs for each job step
# -N1 -n1 allocates a single core to each task - Adapt accordingly
SRUN="srun -n1 --exclusive -c ${SLURM_CPUS_PER_TASK:=1} --cpu-bind=cores"

# /!\ ADAPT TASK variable accordingly
# Absolute path to the (serial) task to be executed i.e. your favorite
# Java/C/C++/Ruby/Perl/Python/R/whatever program to be run
TASK=${TASK:=${HOME}/bin/app.exe}
MIN=1
MAX=30

################################################################################
print_error_and_exit() { echo "*** ERROR *** $*"; exit 1; }
usage() {
    cat <<EOF
NAME
  $(basename $0): Sample launcher for aggregating serial
    (one core) tasks within one node using the Bash & (ampersand), a builtin
    control operator used to fork processes, and the wait command.
  Default TASK: ${TASK}
USAGE
  [sbatch] $0 [-n]  [--min MIN] [--max MAX]
  TASK=/path/to/app.exe [sbatch] $0 [-n] [--min MIN] [--max MAX]

  This will run the following command:
  for i in {${MIN}..${MAX}}; do
     ${SRUN} \${TASK} \$i &
  done
  wait

OPTIONS:
  -n --dry-run: Dry run mode
  --min|--max N: set min/max parameter value
EOF
}
################################################################################
# Check for options
while [ $# -ge 1 ]; do
    case $1 in
        -h | --help) usage; exit 0;;
        -n | --noop | --dry-run) CMD_PREFIX=echo;;
        --min) shift; MIN=$1;;
        --max) shift; MAX=$1;;
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
#################################
for i in $(seq ${MIN} ${MAX}); do
    ${CMD_PREFIX} ${SRUN} ${TASK} ${OPTS} $i &     # <-- Ampersand '&' is key
done
wait  # all the child processes to	finish before terminating	the	parent process; CRUCIAL
##################
end=$(date +%s)
cat <<EOF
### Ending timestamp (s): ${end}"
# Elapsed time (s): $(($end-$start))
EOF
