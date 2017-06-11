#!/bin/bash -l

# Root work directory
ROOT=$HOME/PS2

# Temporary directory in /tmp
TEMP=`mktemp -d`
CONFIG_TARBALL=$ROOT/jcell/config.tgz
PARAM_FILE=$ROOT/jcell/jcell_param

MutationProbMin=10
MutationProbMax=99
MutationProbInc=5

CrossoverProbMin=10
CrossoverProbMax=99
CrossoverProbInc=5

# JCell original configuration file
CFG=cfg/genGAforECC.cfg

# Install Jcell
mkdir -p $ROOT
cd $ROOT
[[ ! -d jcell ]] && tar xzvf /mnt/isilon/projects/ulhpc-tutorials/sequential/jcell.tgz
cd jcell/JCell/bin


# Generate the configuration files
for ((i = $MutationProbMin ; i <= $MutationProbMax ; i+=$MutationProbInc)) do
  for ((j = $CrossoverProbMin ; j <= $CrossoverProbMax ; j+=$CrossoverProbInc)) do
    sed "s/MutationProb.*/MutationProb = 0.$i/" $CFG > $TEMP/${i}_${j}.cfg
    sed -i "s/CrossoverProb.*/CrossoverProb = 0.$j/" $TEMP/${i}_${j}.cfg
  done
done

# Create a tar.gz archives containing all the configuration files
cd $TEMP
tar czvf $CONFIG_TARBALL * > $PARAM_FILE

