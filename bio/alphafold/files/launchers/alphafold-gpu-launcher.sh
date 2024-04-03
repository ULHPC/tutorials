#!/bin/bash -l
#SBATCH --job-name="alphafold-gpu-1t-2g"  # job name
#SBATCH --time=48:00:00                 # max job run time hh:mm:ss
#SBATCH --nodes=1                       # number of nodes
#SBATCH --cpus-per-task=14              # gpu-per-task x 7
#SBATCH --output=%x-%j.log              # job log
#SBATCH --partition=gpu                 # partition
#SBATCH --gpus-per-task=2               # number of gpus per node
#SBATCH --ntasks-per-node=1             #

module load tools/Singularity/3.8.1

# Unified memory can be used to request more than just the total GPU memory for the JAX step in AlphaFold
# - A100 GPU has 80GB memory
# - GPU total memory (80) * XLA_PYTHON_CLIENT_MEM_FRACTION (4.0)
# - XLA_PYTHON_CLIENT_MEM_FRACTION default = 0.9
# This example script has 620 GB of unified memory
# - 80x2 GB from A100 GPU + 460 GB DDR from motherboard
# In case of "RuntimeError: Resource exhausted: Out of memory", add the following variables to the slurm script
#export SINGULARITYENV_TF_FORCE_UNIFIED_MEMORY=1
#export SINGULARITYENV_XLA_PYTHON_CLIENT_MEM_FRACTION=4.0

DATABASE_DIR=/work/projects/bigdata_sets/alphafold2
INPUT_DIR=$SCRATCH/alphafold/fastas
OUTPUT_DIR=$SCRATCH/alphafold/runs_gpu

export SINGULARITY_CACHEDIR=$SCRATCH/singularity-cache
export SINGULARITY_BIND="$INPUT_DIR,$OUTPUT_DIR,$DATABASE_DIR"

export CONTAINER_IMAGE_DIR=/work/projects/singularity/ulhpc

# To list input options type:
# singularity exec --nv $CONTAINER_IMAGE_DIR/alphafold.sif python /app/alphafold/run_alphafold.py --help
# If using GPUs then use the '--nv' flag, i.e. 'singularity exec --nv ...'

singularity exec --nv $CONTAINER_IMAGE_DIR/alphafold-2.3.0.sif python /app/alphafold/run_alphafold.py \
 --data_dir=$DATABASE_DIR \
 --uniref90_database_path=$DATABASE_DIR/uniref90/uniref90.fasta \
 --mgnify_database_path=$DATABASE_DIR/mgnify/mgy_clusters_2022_05.fa \
 --bfd_database_path=$DATABASE_DIR/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
 --uniref30_database_path=$DATABASE_DIR/uniref30/UniRef30_2021_03 \
 --pdb70_database_path=$DATABASE_DIR/pdb70/pdb70 \
 --template_mmcif_dir=$DATABASE_DIR/pdb_mmcif/mmcif_files \
 --obsolete_pdbs_path=$DATABASE_DIR/pdb_mmcif/obsolete.dat \
 --model_preset=monomer \
 --max_template_date=2022-1-1 \
 --db_preset=full_dbs \
 --output_dir=$OUTPUT_DIR \
 --fasta_paths=$INPUT_DIR/test.fasta \
 --use_gpu_relax=TRUE

# GPUs must be available if '--use_gpu_relax' is enabled
