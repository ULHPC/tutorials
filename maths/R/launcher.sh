#!/bin/bash -l
#SBATCH -J Rfuture_foss
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=aurelien.ginolhac@uni.lu
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-00:50:00
#SBATCH -p batch
#SBATCH --qos=qos-batch


echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"


# use version 3.4.4 and load the GNU toolchain
module use /opt/apps/resif/data/devel/default/modules/all
module load lang/R/3.4.4-foss-2018a-X11-20180131-bare

# prevent sub-spawn, 28 cores -> 28 processes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

Rscript tsne.R > job_${SLURM_JOB_NAME}.out 2>&1

