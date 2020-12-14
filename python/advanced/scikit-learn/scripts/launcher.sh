#!/bin/bash -l

#BATCH -p batch           #batch partition 
#SBATCH -J ipy_engines      #job name
#SBATCH -n 10                # 10 cores, you can increase it
#SBATCH -N 2                # 2 node, you can increase it
#SBATCH -t 1:00:00         # Job is killed after 1h

module load lang/Python 

source scikit/bin/activate

#create a new ipython profile appended with the job id number
profile=job_${SLURM_JOB_ID}

echo "Creating profile_${profile}"
ipython profile create ${profile}

ipcontroller --ip="*" --profile=${profile} &
sleep 10

#srun: runs ipengine on each available core
srun ipengine --profile=${profile} --location=$(hostname) &
sleep 25

echo "Launching job for script $1"
python $1 -p ${profile}
