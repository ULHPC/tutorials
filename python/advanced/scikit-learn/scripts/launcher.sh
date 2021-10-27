#!/bin/bash -l

#BATCH -p batch           #batch partition 
#SBATCH -J ipy_engines      #job name
#SBATCH -N 2                # 2 node, you can increase it
#SBATCH -n 10                # 10 task, you can increase it
#SBATCH -c 1                # 1 cpu per task
#SBATCH -t 1:00:00         # Job is killed after 1h

module load lang/Python 

source scikit_${ULHPC_CLUSTER}/bin/activate

#create a new ipython profile appended with the job id number
profile=job_${SLURM_JOB_ID}

echo "Creating profile_${profile}"
ipython profile create ${profile}

# Number of tasks - 1 controller task - 1 python task 
NB_WORKERS=$((${SLURM_NTASKS}-2))

LOG_DIR="$(pwd)/logs/job_${SLURM_JOBID}"
mkdir -p LOG_DIR

#srun: runs ipcontroller -- forces to start on first node 
srun -w $(hostname) --output=${LOG_DIR}/ipcontroller-%j-workers.out  --exclusive -N 1 -n 1 -c ${SLURM_CPUS_PER_TASK} ipcontroller --ip="*" --profile=${profile} &
sleep 10

#srun: runs ipengine on each available core -- controller location first node
srun --output=${LOG_DIR}/ipengine-%j-workers.out --exclusive -n ${NB_WORKERS} -c ${SLURM_CPUS_PER_TASK} ipengine --profile=${profile} --location=$(hostname) &
sleep 25

#srun: starts job
echo "Launching job for script $1"
srun --output=${LOG_DIR}/code-%j-execution.out  --exclusive -N 1 -n 1 -c ${SLURM_CPUS_PER_TASK} python $1 -p ${profile} 

