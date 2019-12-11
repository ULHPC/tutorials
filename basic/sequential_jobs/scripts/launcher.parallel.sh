#!/bin/bash -l
# Time-stamp: <Wed 2019-12-11 14:22 svarrette>
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
#
#SBATCH -J GnuParallel
#SBATCH --time=0-01:00:00     # 1 hour
#SBATCH --partition=batch
#SBATCH --qos qos-batch
#SBATCH -N 1                  # Stick to a single node
#SBATCH --ntasks-per-node 28
### -c, --cpus-per-task=<ncpus>
###     (multithreading) Request that ncpus be allocated per task
###     /!\ Adapt '--ntasks-per-node' above accordingly
#SBATCH -c 1
#SBATCH -o %x-%j.out          # Logfile: <jobname>-<jobid>.out
#
CMD_PREFIX=

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

# Task/app to run - here (by default) a stress command
TASK="stress --cpu ${SLURM_CPUS_PER_TASK:=1} --timeout 60s --vm-hang"

############################################################
print_error_and_exit() { echo "*** ERROR *** $*"; exit 1; }
usage() {
    cat <<EOF
NAME
    $(basename $0) [-n] [TASK]

    Using GNU parallel within a single node to run embarrasingly parallel
    problems, i.e. execute multiple times the command '\${TASK}' within a
    'tunnel' set to run NO MORE THAN \${SLURM_NTASKS} tasks in parallel.

    State of the execution is stored in logs/state.parallel.log and is used to
    resume the execution later on, from where it stoppped (either due to the
    fact that the slurm job has been stopped by failure or by hitting a walltime
    limit) next time you invoke this script.
    In particular, if you need to rerun this GNU Parallel job, be sure to delete
    the logfile logs/state*.parallel.log or it will think it has already
    finished!

    By default, the '${TASK} <arg>' command is executed
    with the arguments {1..8}

OPTIONS
  -n --noop --dry-run:   dry run mode

EXAMPLES
  Within an interactive job (use --exclusive for some reason in that case)
      (access)$> si --exclusive --ntasks-per-node 4
      (node)$> $0 -n    # dry-run
      (node)$> $0
  Within a passive job
      (access)$> sbatch --ntasks-per-node 4 $0
  Within a passive job, using several cores (6) per tasks
      (access)$> sbatch --ntasks-per-socket 2 --ntasks-per-node 4 -c 6 $0

  Get the most interesting usage statistics of your jobs <JOBID> (in particular
  for each job step) with:
     sacct -j <JOBID> --format User,JobID,Jobname,partition,state,time,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist,ConsumedEnergyRaw
EOF
}

####################### Let's go ##############################
# Parse the command-line argument
while [ $# -ge 1 ]; do
    case $1 in
        -h | --help) usage; exit 0;;
        -n | --noop | --dry-run) CMD_PREFIX=echo;;
        *) TASK="$*";;
    esac
    shift;
done

# Use the UL HPC modules
if [ -f  /etc/profile ]; then
    .  /etc/profile
fi
module purge || print_error_and_exit "Unable to find the module command - you're NOT on a computing node"
# module load [...]


#######################
# Create logs directory
mkdir -p logs

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
${CMD_PREFIX} ${PARALLEL} "${SRUN} ${TASK} {1}" ::: {1..8}

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
