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

experiment=/home/users/ppochelu/project/random_search/GNUParallel/exo2_gnu_parallel_multinode/experiment.sh

# Run. The -j option controls how many experiments run in each node (they will share the 30 cores).
# The number of experiments is given by N*j.
parallel  --sshloginfile  $hosts_file  -j 2 $experiment {} ::: a b c d e f g h i j k l m n o p q r s t u v w x y z


