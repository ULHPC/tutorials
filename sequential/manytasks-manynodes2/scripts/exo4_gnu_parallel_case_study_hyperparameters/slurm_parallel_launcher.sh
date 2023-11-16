#!/bin/sh -l
#SBATCH -c 100 # How many cores to use in 1 single node ?
#SBATCH -N 3 # How many nodes ?
#SBATCH -t 5
#SBATCH --export=ALL


# get host name
hosts_file="hosts.txt"
scontrol show hostname $SLURM_JOB_NODELIST > $hosts_file


# Collect public key and accept them
while read -r node; do
    ssh-keyscan "$node" >> ~/.ssh/known_hosts
done < "$hosts_file"


# Store in PARAMS the experiments pool
PARAMS=$(python3 parameters.py)

# Get the Python script
experiment_py=${PWD}/experiment.py

output=${PWD}/output.txt

parallel  --sshloginfile  $hosts_file  -j 100 python3 $experiment_py >> $output  {}  ::: $PARAMS


