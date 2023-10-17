#!/bin/bash -l
#SBATCH -J Distrbuted_gurobi
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=00:15:00
#SBATCH -p batch
#SBATCH --qos normal
#SBATCH -o %x-%j.log

# Explicit cpus-per-task to srun
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# Load personal modules
mu
# Load gurobi
module load math/Gurobi/8.1.1-intel-2018a-Python-3.6.4

export MASTER_PORT=61000
export SLAVE_PORT=61000
export MPS_FILE=$1
export RES_FILE=$2
export GUROBI_INNER_LAUNCHER="inner_job${SLURM_JOBID}.sh"

if [[ -f "grb_rs.cnf" ]];then
    sed -i "s/^THREADLIMIT.*$/THREADLIMIT=${SLURM_CPUS_PER_TASK}/g" grb_rs.cnf
else
    $GUROBI_REMOTE_BIN_PATH/grb_rs init
    echo "THREADLIMIT=${SLURM_CPUS_PER_TASK}" >> grb_rs.cnf
fi


cat << 'EOF' > ${GUROBI_INNER_LAUNCHER}
#!/bin/bash
MASTER_NODE=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
    ## Load configuration and environment
    if [[ ${SLURM_PROCID} -eq 0 ]]; then
        ## Start Gurobi master worker in background
         $GUROBI_REMOTE_BIN_PATH/grb_rs --worker --port ${MASTER_PORT} &
         wait
    elif [[ ${SLURM_PROCID} -eq 1 ]]; then
        sleep 5
        grbcluster nodes --server ${MASTER_NODE}:${MASTER_PORT} 
        gurobi_cl Threads=${SLURM_CPUS_PER_TASK} ResultFile="${RES_FILE}.sol" Workerpool=${MASTER_NODE}:${MASTER_PORT} DistributedMIPJobs=$((SLURM_NNODES -1)) ${MPS_FILE}
    else
        sleep 2
        ## Start Gurobi slave worker in background
        $GUROBI_REMOTE_BIN_PATH/grb_rs --worker --port ${MASTER_PORT} --join ${MASTER_NODE}:${MASTER_PORT} &
        wait
fi
EOF
chmod +x ${GUROBI_INNER_LAUNCHER}

## Launch Gurobi and wait for it to start
srun  ${GUROBI_INNER_LAUNCHER} &
while [[ ! -e "${RES_FILE}.sol" ]]; do
    sleep 5
done
rm ${GUROBI_INNER_LAUNCHER}
