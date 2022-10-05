#!/usr/bin/bash -l

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }


hash parallel 2>/dev/null && test $? -eq 0 || print_error_and_exit "Parallel is not installed on the system"

if [[ -z "${SLURM_NODEID}" ]]; then
    print_error_and_exit "The variable \${SLURM_NODEID} is required (but missing)... exiting"
fi
if [[ -z "${SLURM_NNODES}" ]]; then
    print_error_and_exit "The variable \${SLURM_NNODES} is required (but missing)... exiting"
    exit
fi


NBCORES_TASK=1
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
HOSTNAME=$(hostname)
LOGS_DIR="${SCRIPT_DIR}/logs/Worker${SLURM_NODEID}"
SCRIPT_MAINLOG="${LOGS_DIR}/${SCRIPT_NAME//sh/log}"
RESUME='n'
PARALLEL="parallel --delay 0.2 -j $((SLURM_CPUS_PER_TASK / NBCORES_TASK)) --joblog ${SCRIPT_MAINLOG} "
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
TASKFILE=""
NTASKS=""


#=======================
# Get Optional arguments
#=======================
while [ $# -ge 1 ]; do
    case $1 in
        -r | --resume)           PARALLEL="${PARALLEL} --resume "; RESUME='y';;
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
fi

echo "Starting worker initialisation on $(hostname)"

# Every worker clean its own log directory
if [[ -d ${LOGS_DIR} ]] && [[ ${RESUME} == 'n' ]];then
    echo "Create archive from old results  ${LOGS_DIR}"
    tar -zcvf "${LOGS_DIR}-${TIMESTAMP}.tar.gz" ${LOGS_DIR}
    echo "Cleaning ${LOGS_DIR}"
    find ${LOGS_DIR} -mindepth 1 -print -delete
fi

echo "Create logs directory if not existing"
mkdir -p ${LOGS_DIR}

if [[ -z ${NTASKS} ]];then
    cat ${TASKFILE} |                                      \
    awk -v NNODE="$SLURM_NNODES" -v NODEID="$SLURM_NODEID" \
    'NR % NNODE == NODEID' |                               \
    ${PARALLEL} "{1} > ${LOGS_DIR}/$(basename ${TASKFILE}).log.{%}"
else
    echo  "$(seq 1 ${NTASKS})" |                             \
    awk -v NNODE="$SLURM_NNODES" -v NODEID="$SLURM_NODEID" \
    'NR % NNODE == NODEID' |                               \
    ${PARALLEL} "${TASKFILE} > ${LOGS_DIR}/$(basename ${TASKFILE}).log.{1}"
fi

