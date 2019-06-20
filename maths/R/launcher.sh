#!/bin/bash -l
#SBATCH -J tsne_hpc
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-00:10:00
#SBATCH -p batch
#SBATCH --qos=qos-batch

echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"

# use version 3.6.0 and load the GNU toolchain
module load swenv/default-env/devel
module load lang/R/3.6.0-foss-2019a-bare

# prevent sub-spawn, 28 cores -> 28 processes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

Rscript tsne.R > job_${SLURM_JOB_NAME}.out 2>&1

