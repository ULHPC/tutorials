#!/bin/bash -l
# Time-stamp: <Sun 2017-06-11 21:18 svarrette>
##################################################################
# Usage: $0 [--eth]    Run OSU benchmarks with OpenMPI suite
#        $0 intel      Run OSU benchmarks with Intel MPI suite
#        $0 mvapich    Run OSU benchmarks with MVAPICH2 suite
##################################################################
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
### -c, --cpus-per-task=<ncpus>
###     Request that ncpus be allocated per process
#SBATCH -c 1
#SBATCH --partition=batch
#SBATCH --time=00:10:00
#SBATCH --job-name=OSU-Microbenchmark
#SBATCH --qos qos-batch

### General SLURM Parameters
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "Submission directory = "$SLURM_SUBMIT_DIR

### Default mode
MODE='openmpi'
MCA="--mca btl openib,self,sm"
case $1 in
    --eth*) MCA="--mca btl tcp,self";;
    intel*   | --intel*)   MODE='intel';;
    mvapich* | --mvapich*) MODE='mvapich2';;
    *) MODE='openmpi';;
esac

### Prepare the output directory
SCRIPTFILENAME=$(basename $0)
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    echo "toto SCRIPTDIR=${SCRIPTDIR}"
    [[ "${SCRIPTDIR}" == *"slurmd"* ]] && RUNDIR=${SLURM_SUBMIT_DIR} || RUNDIR=${SCRIPTDIR}
else
    RUNDIR=${SCRIPTDIR}
fi
echo "RUNDIR=${RUNDIR}"
TOP_SRCDIR="$( cd ${RUNDIR}/.. && pwd )"
[ -n "${SLURM_JOBID}" ] && JOBDIR="job-${SLURM_JOBID}" || JOBDIR="$(date +%Hh%Mm%S)"
DATADIR="${RUNDIR}/${MODE}/$(date +%Y-%m-%d)/${JOBDIR}"
BUILDDIR="${TOP_SRCDIR}/build.${MODE}"
APPDIR="${BUILDDIR}/mpi/one-sided"
# Delay between each run
DELAY=1

### Toolbox function
print_error_and_exit() {
    echo "***ERROR*** $*"
    exit 1
}

############################################
################# Let's go #################
############################################

# Use the RESIF build modules
if [ -f  /etc/profile ]; then
   .  /etc/profile
fi

# Default MPI command
MPI_CMD="srun -n $SLURM_NTASKS"
# Module load and appropriate MPI command
case $MODE in
    intel)
        module load toolchain/intel;;
    mvapich2)
        module load mpi/MVAPICH2;;
    *)
        # Assumes OpenMPI by default
        module load mpi/OpenMPI;
        MPI_CMD="mpirun -np $SLURM_NTASKS -npernode 1 ${MCA}";;
esac

[ ! -d "${APPDIR}" ] && print_error_and_exit "Unable to find the application directory '${APPDIR}'"
if [ ! -d "${DATADIR}" ]; then
    echo "=> creating ${DATADIR}"
    mkdir -p ${DATADIR}
fi

for prog in osu_get_latency osu_get_bw; do
    logfile="${DATADIR}/${prog}_$(date +%Hh%Mm%S).log"
    mpi_cmd="${MPI_CMD} ${APPDIR}/$prog"
    echo "=> performing MPI OSU benchmark '${prog}' @ $(date) using:"
    echo "   ${mpi_cmd}"
        cat > ${logfile} <<EOF
# ${logfile}
# MPI Run '${prog}', Generated @ $(date) by:
#   $mpi_cmd
#
# SLURM_JOBID        $SLURM_JOBID
# SLURM_JOB_NODELIST $SLURM_JOB_NODELIST
# SLURM_NNODES       $SLURM_NNODES
# SLURM_NTASKS       $SLURM_NTASKS
# SLURM_SUBMIT_DIR   $SLURM_SUBMIT_DIR
### Starting timestamp: $(date +%s)
EOF
    cd ${DATADIR}
    echo "   command performed in $(pwd)"
    $mpi_cmd |& tee -a ${logfile}
    cd -
    cat >> ${logfile} <<EOF
### Ending timestamp:     $(date +%s)
EOF
    echo "=> now sleeping for ${DELAY}s"
    sleep $DELAY
done
