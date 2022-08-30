#!/bin/bash -l
# Time-stamp: <Mon 2021-11-15 21:43 svarrette>
################################################################################
# Slurm launcher for embarrassingly parallel problems combining srun and GNU
# parallel within a single node to runs multiple times the command ${TASK}
# within a 'tunnel' set to execute no more than ${SLURM_NTASKS} tasks in
# parallel.
#
# Resources:
# - https://www.marcc.jhu.edu/getting-started/additional-resources/distributing-tasks-with-slurm-and-gnu-parallel/
# - https://rcc.uchicago.edu/docs/tutorials/kicp-tutorials/running-jobs.html
# - https://curc.readthedocs.io/en/latest/software/GNUParallel.html
#
# To use this script, simply adapt the definition of the variables:
#   TASK: path to the [serial] task/application to run
#   TASKLIST[FILE] : input list of tasks arguments
#################################################################################
#SBATCH -J GnuParallel
###SBATCH --dependency singleton
###SBATCH --mail-type=FAIL     # Mail events (NONE, BEGIN, END, FAIL, ALL)
###SBATCH --mail-user=<email>
#SBATCH --time=0-01:00:00      # 1 hour
#SBATCH --partition=batch
#__________________________
#SBATCH -N 1
#SBATCH --ntasks-per-node 28   # Optimized for 1 full node of iris
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
PARALLEL="parallel"

### GNU Parallel options
# Logfile used to save a list of the executed job for executed job.
# combines with --resume to resume from the last unfinished job.
JOBLOGFILE=logs/state.parallel.log

### /!\ ADAPT TASK and TASKLIST[FILE] variables accordingly
# Absolute path to the (serial) task to be executed i.e. your favorite
# Java/C/C++/Ruby/Perl/Python/R/whatever program to be run
TASK=${TASK:=${HOME}/bin/app.exe}
# Default task list, here {1..8} is a shell expansion interpreted as 1 2... 8
TASKLIST="{1..8}"
# TASKLISTFILE=path/to/input_file
# If non-empty, used as input source over TASKLIST.
TASKLISTFILE=

############################################################
print_error_and_exit() { echo "*** ERROR *** $*"; exit 1; }
usage() {
    less <<EOF
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
    Using '-a <input_file>', <input_file> is used as input source

USAGE
   [sbatch] $0 [-n] [TASKLIST]
   TASK=/path/to/app.exe [sbatch] $0 [-n] [TASKLIST]

   [sbatch] $0 [-n] -a TASKLISTFILE
   TASK=/path/to/app.exe [sbatch] $0 [-n] -a TASKLISTFILE

OPTIONS
  -a --arg-file FILE  Use FILE as input source
  -d --debug          Print debugging information
  --joblog FILE       Logfile for executed jobs (Default: ${JOBLOGFILE})
  -n --dry-run        Dry run mode (echo **full** parallel command)
  -t --test --noop    No-operation mode: echo the run commands

EXAMPLES
  Within an interactive job (use --exclusive for some reason in that case)
      (access)$> si --ntasks-per-node 28
      (node)$> $0 -n    # dry-run - print full parallel command
      (node)$> $0 -t    # noop mode - print commands
      (node)$> $0
  Within a passive job
      (access)$> sbatch $0
  Within a passive job, using several cores (2) per tasks
      (access)$> sbatch --ntasks-per-socket 7 --ntasks-per-node 14 -c 2 $0
  Use another range of parameters - don't forget the souble quotes
      (access)$> sbatch $0 "{1..100}"
  Use an input file
      (access)$> sbatch $0 -a path/to/tasklistfile

  Get the most interesting usage statistics of your jobs <JOBID> (in particular
  for each job step) with:
     slist <JOBID> [-X]

AUTHOR
  S. Varrette and UL HPC Team <hpc-team@uni.lu>

COPYRIGHT
	This is free software; see the source for copying conditions. There is NO
	warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
EOF
}
print_debug_info(){
cat <<EOF
 TOP_DIR      = ${TOP_DIR}
 PARALLEL     = ${PARALLEL}
 TASK         = ${TASK}
 TASKLIST     = $(eval echo ${TASKLIST})
 TASKLISTFILE = ${TASKLISTFILE}
EOF
[ -n "${SLURM_JOBID}" ] && echo "$(scontrol show job ${SLURM_JOBID})"
}
####################### Let's go ##############################
# Check for options
while [ $# -ge 1 ]; do
    case $1 in
        -h | --help) usage; exit 0;;
        -d | --debug)    DEBUG=$1;;
        -a | --arg-file) shift;
            TASKLISTFILE=$1;
            PARALLEL="${PARALLEL} -a $1";;
        --joblog) shift; JOBLOGFILE=$1;;
        -n | --dry-run)  CMD_PREFIX=echo;;
        -t | --test | --noop) ECHO=echo;;
        *) TASKLIST=$*;;
    esac
    shift;
done
### GNU Parallel command
# --delay .2 prevents overloading the controlling node
# -j is the number of tasks parallel runs so we set it to $SLURM_NTASKS
# --joblog makes parallel create a log of tasks that it has already run
# --resume makes parallel use the joblog to resume from where it has left off
#   the combination of --joblog and --resume allow jobs to be resubmitted if
#   necessary and continue from where they left off
PARALLEL="${PARALLEL} --delay .2 -j ${SLURM_NTASKS} --joblog ${JOBLOGFILE} --resume"
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
# GNU Parallel syntax:
#   Reading command arguments on the command line:
#         parallel [OPTIONS] COMMAND {} ::: TASKLIST
#   Reading command arguments from an input file (prefer the first syntax to avoid confusion)
#         parallel –a TASKLISTFILE [OPTIONS] COMMAND {}
#         parallel [OPTIONS] COMMAND {} :::: TASKLISTFILE
if [ -z "${TASKLISTFILE}" ]; then
    # parallel uses ::: to separate options. Here default {1..8} is a shell expansion
    # so parallel will run the command passing the numbers 1 through 8 via argument {1}
    ${CMD_PREFIX} ${PARALLEL} "${ECHO} ${SRUN} ${TASK} {1}" ::: $(eval echo ${TASKLIST})
else
    # use ${TASKLISTFILE} as input source for TASKLIST
    ${CMD_PREFIX} ${PARALLEL} -a ${TASKLISTFILE} "${ECHO} ${SRUN} ${TASK} {}"
fi

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
    echo "/!\ WARNING: Test mode - removing joblog state file '${JOBLOGFILE}'"
    rm -f ${JOBLOGFILE}
fi
