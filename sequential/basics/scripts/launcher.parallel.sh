#!/bin/bash -l
# Time-stamp: <Fri 2020-12-11 14:55 svarrette>
#############################################################################
# Slurm launcher for embarrassingly parallel problems combining srun and GNU
# parallel within a single node to runs multiple times the command ${TASK}
# within a 'tunnel' set to execute no more than ${SLURM_NTASKS} tasks in
# parallel.
#
# Resources:
# - https://www.marcc.jhu.edu/getting-started/additional-resources/distributing-tasks-with-slurm-and-gnu-parallel/
# - https://rcc.uchicago.edu/docs/tutorials/kicp-tutorials/running-jobs.html
# - https://curc.readthedocs.io/en/latest/software/GNUParallel.html
#############################################################################
#SBATCH -J GnuParallel
###SBATCH --dependency singleton
###SBATCH --mail-type=FAIL     # Mail events (NONE, BEGIN, END, FAIL, ALL)
###SBATCH --mail-user=<email>
#SBATCH --time=0-01:00:00      # 1 hour
#SBATCH --partition=batch
#__________________________
#SBATCH -N 1
#SBATCH --ntasks-per-node 28
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
ECHO=
DEBUG=
# the --exclusive to srun makes srun use distinct CPUs for each job step
# -N1 -n1 allocates a single core to each task - Adapt accordingly
SRUN="srun  --exclusive -n1 -c ${SLURM_CPUS_PER_TASK:=1} --cpu-bind=cores"

### GNU Parallel options
# --delay .2 prevents overloading the controlling node
# -j is the number of tasks parallel runs so we set it to $SLURM_NTASKS
# --joblog makes parallel create a log of tasks that it has already run
# --resume makes parallel use the joblog to resume from where it has left off
#   the combination of --joblog and --resume allow jobs to be resubmitted if
#   necessary and continue from where they left off
PARALLEL="parallel --delay .2 -j ${SLURM_NTASKS} --joblog logs/state.parallel.log --resume"

# /!\ ADAPT TASK and TASKLIST variables accordingly
# Absolute path to the (serial) task to be executed i.e. your favorite
# Java/C/C++/Ruby/Perl/Python/R/whatever program to be run
TASK=${TASK:=${HOME}/bin/app.exe}
TASKLIST="{1..8}"

############################################################
print_error_and_exit() { echo "*** ERROR *** $*"; exit 1; }
usage() {
    cat <<EOF
NAME
    $(basename $0): Generic launcher using GNU parallel
    within a single node to run embarrasingly parallel problems, i.e. execute
    multiple times the command '\${TASK}' within a 'tunnel' set to run NO MORE
    THAN \${SLURM_NTASKS} tasks in parallel.
    State of the execution is stored in logs/state.parallel.log and is used to
    resume the execution later on, from where it stoppped (either due to the
    fact that the slurm job has been stopped by failure or by hitting a walltime
    limit) next time you invoke this script.
    In particular, if you need to rerun this GNU Parallel job, be sure to delete
    the logfile logs/state*.parallel.log or it will think it has already
    finished!
    By default, '${TASK}' command is executed with arguments ${TASKLIST}

USAGE
   [sbatch] $0 [-n] [TASKLIST]
   TASK=/path/to/app.exe [sbatch] $0 [-n] [TASKLIST]
OPTIONS
  -n --dry-run:      dry run mode (echo full parallel command)
  -t --test --noop:  no-operation mode: echo run commands

EXAMPLES
  Within an interactive job (use --exclusive for some reason in that case)
      (access)$> si --ntasks-per-node 28
      (node)$> $0 -n    # dry-run
      (node)$> $0
  Within a passive job
      (access)$> sbatch $0
  Within a passive job, using several cores (2) per tasks
      (access)$> sbatch --ntasks-per-socket 7 --ntasks-per-node 14 -c 2 $0

  Get the most interesting usage statistics of your jobs <JOBID> (in particular
  for each job step) with:
     slist <JOBID> [-X]
EOF
}
print_debug_info(){
cat <<EOF
 TOP_DIR    = ${TOP_DIR}
 TASK       = ${TASK}
 TASKLIST   = $(eval echo ${TASKLIST})
EOF
[ -n "${SLURM_JOBID}" ] && echo "$(scontrol show job ${SLURM_JOBID})"
}
####################### Let's go ##############################
# Check for options
while [ $# -ge 1 ]; do
    case $1 in
        -h | --help) usage; exit 0;;
        -d | --debug)   DEBUG=$1;;
        -n | --dry-run) CMD_PREFIX=echo;;
        -t | --test | --noop) ECHO=echo;;
        *) TASKLIST=$*;;
    esac
    shift;
done
[ -n "${DEBUG}"  ] && print_debug_info
[ ! -x "${TASK}" ] && print_error_and_exit "Unable to find TASK=${TASK}"
module purge || print_error_and_exit "Unable to find the 'module' command"
### module load [...]
# module load lang/Python
# source ~/venv/<name>/bin/activate

start=$(date +%s)
echo "### Starting timestamp (s): ${start}"

# this runs the parallel command you want, i.e. running the
# script ${TASK} within a 'tunnel' set to run no more than ${SLURM_NTASKS} tasks
# in parallel
# See 'man parallel'
# - Reader's guide: https://www.gnu.org/software/parallel/parallel_tutorial.html
# - Numerous (easier) tutorials are available online. Ex:
#   http://www.shakthimaan.com/posts/2014/11/27/gnu-parallel/news.html
#
# parallel uses ::: to separate options. Here {1..8} is a shell expansion
# so parallel will run the command passing the numbers 1 through 8 via argument {1}
${CMD_PREFIX} ${PARALLEL} "${ECHO} ${SRUN} ${TASK} {1}" ::: $(eval echo ${TASKLIST})

end=$(date +%s)
cat <<EOF
### Ending timestamp (s): ${end}"
# Elapsed time (s): $(($end-$start))

Beware that the GNU parallel option --resume makes it read the log file set by
--joblog (i.e. logs/state*.log) to figure out the last unfinished task (due to the
fact that the slurm job has been stopped due to failure or by hitting a walltime
limit) and continue from there.
In particular, if you need to rerun this GNU Parallel job, be sure to delete the
logfile logs/state*.parallel.log or it will think it has already finished!
EOF

# If in test mode, remove the state file
if [ -n "${ECHO}" ]; then
    echo "/!\ WARNING: Test mode - removing sate file"
    rm -f logs/state.parallel.log
fi
