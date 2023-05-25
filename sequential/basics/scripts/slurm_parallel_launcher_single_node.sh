#!/bin/bash -l
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1


# Just useful check
print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
hash parallel 2>/dev/null && test $? -eq 0 || print_error_and_exit "Parallel is not installed on the system"

# Increase the user process limit.
ulimit -u 10000

echo "Node: ${SLURM_NODELIST}"
echo "Executing ${SLURM_NTASK} independant tasks at the same time"
export TIMESTAMP=$(date +"%Y%m%dT%H%M%S")


# the --exclusive to srun makes srun use distinct CPUs for each job step
# -N1 -n1 single task with ${SLURM_CPUS_PER_TASK} cores
SRUN="srun  --exclusive -n1 -c ${SLURM_CPUS_PER_TASK:=1} --cpu-bind=cores"

HOSTNAME=$(hostname)
LOGS="logs.${TIMESTAMP}"
RESUME=""
TASKFILE=""
NTASKS=""


#=======================
# Get Optional arguments
#=======================
while [ $# -ge 1 ]; do
    case $1 in
        -r | --resume)           shift; LOGS=$1; RESUME=" --resume ";;
        -n | --ntasks)           shift; NTASKS="$1"                            ;;
        -* | --*)                echo "[Warning] Invalid option $1"          ;;
        *)                       break                                       ;;
    esac
    shift
done

#=======================
# Get Mandatory  Options
#=======================

if [[ "$#" < 1 ]]; then
    print_error_and_exit "No tasks defined"
else
    TASKFILE="$1"
    TASKFILE_DIR=$(cd "$(dirname "${TASKFILE}")" && pwd)
    TASKFILE_NAME="$(basename "${TASKFILE}")"
fi

echo "Starting parallel worker initialisation on $(hostname)"

#=======================
# Set logs for resuming
#=======================

LOGS_DIR="${TASKFILE_DIR}/${LOGS}"
TASKFILE_MAINLOG="${LOGS_DIR}/${TASKFILE_NAME//sh/log}"
PARALLEL="parallel --delay 0.2 -j ${SLURM_NTASKS} --joblog ${TASKFILE_MAINLOG} ${RESUME}"


echo "Create logs directory if not existing"
mkdir -p ${LOGS_DIR}

if [[ -z ${NTASKS} ]];then
    cat ${TASKFILE} |                                      \
    awk -v NNODE="$SLURM_NNODES" -v NODEID="$SLURM_NODEID" \
    'NR % NNODE == NODEID' |                               \
    ${PARALLEL} "${SRUN} {1} > ${LOGS_DIR}/$(basename ${TASKFILE}).log.{%}"
else
    echo  "$(seq 1 ${NTASKS})" |                             \
    awk -v NNODE="$SLURM_NNODES" -v NODEID="$SLURM_NODEID" \
    'NR % NNODE == NODEID' |                               \
    ${PARALLEL} "${SRUN} ${TASKFILE} > ${LOGS_DIR}/$(basename ${TASKFILE}).log.{1}"
fi
