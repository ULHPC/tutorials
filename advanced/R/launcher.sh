#!/bin/bash -l
#SBATCH -J Rfuture
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=aurelien.ginolhac@uni.lu
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-03:00:00
##SBATCH -e job.out
##SBATCH -o job.out
#SBATCH -p batch
#SBATCH --qos=qos-batch


echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"

module use /opt/apps/resif/data/devel/default/modules/all
module load lang/R/3.4.4-intel-2018a-X11-20180131-bare
#Rscript ... | tee -a job.out
Rscript tsne.R >job.out 2>&1


