#! /bin/bash -l
# Time-stamp: <Sun 2018-06-10 18:15 svarrette>
######## OAR directives ########
#OAR -n OpenMP
#OAR -l nodes=1/core=4,walltime=0:05:00
#OAR -O OpenMP-%jobid%.log
#OAR -E OpenMP-%jobid%.log
#
####### Slurm directives #######
#SBATCH -J OpenMP
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --time=0-00:05:00
#SBATCH -p batch
#SBATCH --qos=qos-batch
#
### Usage:
# $0 intel  [app]
# $0 foss   [app]

################################################################################
print_error_and_exit() { echo "*** ERROR *** $*"; exit 1; }
usage() {
    cat <<EOF
NAME
  $0 -- OpenMP launcher example
USAGE
  $0 {intel | foss } [app]

Example:
  $0                          run foss on hello_openmp
  $0 intel                    run intel on hello_openmp
  $0 foss matrix_mult_openmp  run foss  on matrix_mult_openmp
EOF
}

################################################################################
# OpenMP Setup
if [ -n "$OAR_NODEFILE" ]; then
    export OMP_NUM_THREADS=$(cat $OAR_NODEFILE| wc -l)
elif [ -n "SLURM_CPUS_PER_TASKS" ]; then
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
fi
# Use the UL HPC modules
if [ -f  /etc/profile ]; then
   .  /etc/profile
fi

################################################################################
# Directory holding your built applications
# /!\ ADAPT to match your own settings
APPDIR="$HOME/tutorials/OpenMP-MPI/basics/bin"
[ -n "$2" ] && APP=$2 || APP=hello_openmp
# Eventual options of the OpenMP program
OPTS=

################################################################################
case $1 in
    -h       | --help)     usage; exit 0;;
    intel*   | --intel*)   APP=intel_${APP}; MODULE=toolchain/intel;;
    *)                     MODULE=toolchain/foss;;
esac

EXE="${APPDIR}/${APP}"
[ ! -x "${EXE}" ]  && print_error_and_exit "Unable to find the generated executable ${EXE}"

echo "# =============================================================="
echo "# => OpenMP run of '${APP}' with ${MODULE}"
echo "#    OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "# =============================================================="

module purge || print_error_and_exit "Unable to find the module command"
module load ${MODULE}
module list

echo "=> running command: ${EXE} ${OPTS}"
${EXE} ${OPTS}
