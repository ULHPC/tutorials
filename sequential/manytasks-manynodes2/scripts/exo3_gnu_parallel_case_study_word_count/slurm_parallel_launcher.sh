#!/bin/sh -l
#SBATCH -c 30 # How many cores to use in 1 single node ?
#SBATCH -N 3 # How many nodes ?
#SBATCH -t 1
#SBATCH --export=ALL


# get host name
hosts_file="hosts.txt"
scontrol show hostname $SLURM_JOB_NODELIST > $hosts_file


# Collect public key and accept them
while read -r node; do
    ssh-keyscan "$node" >> ~/.ssh/known_hosts
done < "$hosts_file"

experiment=${PWD}/experiment.sh
outfile=${PWD}/output.txt

parallel  --sshloginfile  $hosts_file  -j 2 $experiment {} $outfile ::: ${PWD}/input/*.txt


