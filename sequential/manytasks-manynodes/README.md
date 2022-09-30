[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/sequential/basics/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/sequential/manytasks-manynodes/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/sequential/manytasks-manynodes/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Many Tasks — Many Node Allocations using Slurm and GNU Parallel

     Copyright (c) 2022-2023 E. Kieffer and UL HPC Team <hpc-team@uni.lu>

## Going beyond a single node

[GNU Parallel](/sequential/gnu-parallel/) is definitely a powerful system when combined with the [Slurm scheduler](https://slurm.schedmd.com/) and avoid all the disavantages of [Job Arrays](https://slurm.schedmd.com/job_array.html).
An extensive use of job arrays is **NOT** recommanded as described in [HPC Management of Sequential and Embarrassingly Parallel Jobs](/sequential/basic/) while using a script to submit jobs inside a loop is even worse as all your submitted jobs compete to be scheduled.

Although using Parallel on a single node allows to schedule your tasks inside your own allocation, it may not be sufficient. You may want to use a multiple node allocation to run more than **28** independent tasks on the Iris cluster and **128** on the Aion cluster. In mutiple research area, **repeatbility** is necessary and this is absolutely **NOT** embarrassing. In fact, task independence is a perfect case with no communication overhead.

In the [HPC Management of Sequential and Embarrassingly Parallel Jobs](/sequential/basic/) tutorial, you have seen how to use efficiently the [GNU Parallel](/sequential/gnu-parallel/) for executing multiple independent tasks in parallel on a single machine/node. In this tutorial, we will consider a multi-node allocation with GNU Parallel

## Prerequisites

It's very important to make sure that parallel is installed on the computing nodes and not only on the access. On the ULHPC clusters, this should the case. If you would like to test it on a different platform, please refer to [GNU Parallel](/sequential/gnu-parallel/) to learn how to install it locally. Some other  well-know HPC centers provides GNU Parallel as a [Lmod](https://lmod.readthedocs.io/en/latest/) module.

## Methodology


## Spawning parallel workers

By default, GNU Parallel can distribute tasks to multiples nodes using the `--sshlogin` feature. Nonetheless, the approach is not really effective as suggested in the [NERSC documentation](https://docs.nersc.gov/jobs/workflow/gnuparallel/#scaling-parallel-with-sshlogin-is-not-recommended). In fact, we will use a simple and efficient approach to distribute tasks to nodes in a round-robin manner.  

Using `srun`, we will spawn a single parallel worker on each node. Each parallel worker will manage its own queue of tasks which implies that if a node fail only its queue will be impacted. To do so, have a look on the `slurm_parallel_launcher.sh` script:

```bash
#!/bin/bash -l
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --nodes=10
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

# Increase the user process limit.
ulimit -u 10000

echo "Spawning ${SLURM_NTASKS_PER_NODE} parallel worker on ${SLURM_NNODES} nodes"
echo "Nodes: ${SLURM_NODELIST}"
echo "Each parallel worker can execute ${SLURM_CPUS_PER_TASK} independant tasks" 
srun --no-kill --wait=0 parallel_worker.sh $*
```
* The option `--exclusive` has been added to be sure that we do not share the node with another job. 
* The option `--mem=0` ensures that all memory will be available
* **Note**: Even if the previous options were missing, the `--nstasks-per-node` and `--cpus-per-task` options would ensure that all memory and cores allocated. In some situation, you may not need all cores but all the memory. Therefore I keep `--mem=0` and `--exclusive`. 

You probably notice the srun options `--no-kill` and `--wait=0` which respectively ensure that the job does **NOT** automatically terminate  if one of the nodes it has been allocated fails and does wait on all tasks, i.e., does not terminate some time after the first task has been completed.

## The parallel worker

We now need to define the script starting the parallel workers `parallel_worker.sh`. Please keep in mind that this script will be executing exactly ${SLURM_NTASKS} times. Since we want a single parallel worker on each node, we have to specify in the previous launcher `--ntasks-per-node=1`. 


```bash
#!/bin/bash -l

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }


hash parallel 2>/dev/null && test $? -eq 0 || print_error_and_exit "Parallel is not installed on the system"

if [[ -z "${SLURM_NODEID}" ]]; then
    print_error_and_exit "The variable \${SLURM_NODEID} is required (but missing)... exiting"
fi
if [[ -z "${SLURM_NNODES}" ]]; then
    print_error_and_exit "The variable \${SLURM_NNODES} is required (but missing)... exiting"
    exit
fi


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
HOSTNAME=$(hostname)
LOGS_DIR="${SCRIPT_DIR}/logs/Worker${SLURM_NODEID}"
SCRIPT_MAINLOG="${LOGS_DIR}/${SCRIPT_NAME//sh/log}"
RESUME='n'
PARALLEL="parallel --delay 0.2 -j ${SLURM_CPUS_PER_TASK} --joblog ${SCRIPT_MAINLOG} "
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
TASKFILE=""
NTASKS=""


echo "$#"
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
```

## Exercices






