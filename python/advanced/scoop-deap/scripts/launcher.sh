#!/bin/bash -l


# Ensure process affinity is disabled
export SLURM_CPU_BIND=none

# Prepare in the current folder a worker launcher for Scoop 
# The scipt below will 'decorate' the python interpreter command
# Before python is called, modules are loaded
HOSTFILE=$(pwd)/hostfile
SCOOP_WRAPPER=$(pwd)/scoop-python.sh

cat << EOF > $SCOOP_WRAPPER
#!/bin/bash -l
module load lang/Python
export SLURM_NTASKS=${SLURM_NTASKS}
source $(pwd)/scoop_env_${ULHPC_CLUSTER}/bin/activate
EOF
echo 'python $@' >> $SCOOP_WRAPPER

chmod +x $SCOOP_WRAPPER

# Classical "module load" in the main script
module load lang/Python
source $(pwd)/scoop_env_${ULHPC_CLUSTER}/bin/activate

# Save the hostname of the allocated nodes
scontrol show hostnames > $HOSTFILE

# Start scoop with python input script
INPUTFILE=$(pwd)/evolution.py 
python -m scoop --hostfile $HOSTFILE -n ${SLURM_NTASKS} --python-interpreter=$SCOOP_WRAPPER $INPUTFILE $@
