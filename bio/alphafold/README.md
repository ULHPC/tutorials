[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/bio/basics/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/bio/basics/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/bio/basics/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Alphafold on the UL HPC platform

Copyright (c) 2024 UL HPC Team  <hpc-sysadmins@uni.lu>

Authors: Arnaud Glad

The objective of this tutorial is to explain how to use Alphafold on the [UL HPC](http://hpc.uni.lu) platform.

This tutorial will:

1. show you how to launch a sample experiment with Alphafold and access the results
2. show you how to customize the launcher scripts for your experiements
<!--3. have a look at performance analysis and help you choose on which hardware (CPU or GPU) to run your experiment and evaluate how much resources to request-->

## Prerequisites

This tutorial supposes you are comfortable using the [command line interface](https://ulhpc-tutorials.readthedocs.io/en/latest/linux-shell/) (i.e. manipulate files and directories) and [submit jobs](https://ulhpc-tutorials.readthedocs.io/en/latest/basic/scheduling/) to the cluster.

### Get the tutorial files

The tutorial files and the launcher are available on the UL HPC git repository. Connect to the cluster and clone the tutorial repository:

	[clusteruser@access1 ~]$ git clone https://github.com/ULHPC/tutorials.git
	[clusteruser@access1 ~]$ ln -s tutorials/bio/alphafold/files ~/alphafold
	[clusteruser@access1 ~]$ cd alphafold

Note: The second line creates a shortcut to the tutorial files for more convenient access.

You can list the files using the _tree_ command:

	[clusteruser@access1 alphafold]$ tree -f
	.
	├── ./fasta
	│   └── ./fasta/test.fasta
	└── ./launchers
	    ├── ./launchers/alphafold-cpu-launcher.sh
	    └── ./launchers/alphafold-gpu-launcher.sh


- The fasta directory contains a sample sequence that we will use in the tutorial
- the launchers directory contain slurm launchers targeting either aion for CPU processing or iris for GPU processing.

### Check the Alphafold Singularity image

UL HPC provides Alphafold as a Singularity image. Those images are immutable and provide stable containerized  and independent environment each time they are started. As those images never change, they are great for reproducibility as well as versioning. 

At the time of writing, we propose the 2.3.0 Alphafold version. You can build other versions yourself or request the HLST to do it for you by opening a ticket on service now. [Official releases](https://github.com/google-deepmind/alphafold/releases) are listed on the Alphafold github repository.

The images we build are automatically published on the cluster. You can see all the available images with the following command:

	[clusteruser@access1 alphafold]$ ls /work/projects/singularity/ulhpc/alphafold*
	alphafold-2.3.0.sif

You will have to update the launcher files to use other Alphafold versions.

### Check the the database files

The database files are stored inside a project that is publicly available. 
The models contained in the different versions of Alphafold have been trained to operate using specific version of databases. The 2.3.0 version of Alphafold provided on the cluster works with the databases provided in _/work/projects/bigdata_sets/alphafold2_. 

If you plan to use other versions of Alphafold, please be sure to update the launcher files to point to the relevant database files. If the files required by your version are not present, please open a ticket on service now.

You can find more information about the databases in the official Alphafold repository in the [genetic databases](https://github.com/google-deepmind/alphafold?tab=readme-ov-file#genetic-databases) section.

## Step by step sample experiment

In this section, we show you how to prepare, start and review a small sample experiment.

### Connect to the cluster

Alphafold comes in two flavors on the UL HPC, with CPU launcher and a GPU launcher. 
As GPUs are in high demand (i.e. potentially long wait times before a job starts) and their effectiveness is very dependent on the workload (only marginal gains on this exmaple), we will use the CPU version for this example.

We recommend running the CPU version on aion as the cpus are faster than iris's and the nodes have more memory (256GB on aion vs 128BG on iris).

Connect to the aion cluster using your favorite terminal emulator.

	[localuser@localmachine]$ ssh aion-cluster
	================================================================================
	  Welcome to access1.aion-cluster.uni.lux
	  WARNING: For use by authorized users only!
	            __  _    _                ____ _           _          __
	           / / / \  (_) ___  _ __    / ___| |_   _ ___| |_ ___ _ _\ \
	          | | / _ \ | |/ _ \| '_ \  | |   | | | | / __| __/ _ \ '__| |
	          | |/ ___ \| | (_) | | | | | |___| | |_| \__ \ ||  __/ |  | |
	          | /_/   \_\_|\___/|_| |_|  \____|_|\__,_|___/\__\___|_|  | |
	           \_\                                                    /_/
	================================================================================
	[clusteruser@access1 ~]$           

If you followed the _Prerequisites_ section, you should see an _alphafold_ directory in your home directory that contains a sample fasta (i.e. input file) and the launchers. Move to this folder and have a look at the cpu launcher file.

	[clusteruser@access1 ~]$ cd alphafold/launchers
	[clusteruser@access1 launchers]$ ls
	alphafold-cpu-launcher.sh  alphafold-gpu-launcher.sh
	[clusteruser@access1 launchers]$ cat -n alphafold-cpu-launcher.sh
	 1	#!/bin/bash -l
     2	#SBATCH --job-name="alphafold-cpu-1t-128c"    # job name
     3	#SBATCH --partition=batch           # partition
     4	#SBATCH --time=48:00:00             # max job run time hh:mm:ss
     5	#SBATCH --nodes=1                   # number of nodes
     6	#SBATCH --ntasks-per-node=1         # tasks per compute node
     7	#SBATCH --output=%x-%j.log          # job log
     8	#SBATCH --cpus-per-task=128
     9	
    10
    11	module load tools/Singularity/3.8.1
    12	
    13	DATABASE_DIR=/work/projects/bigdata_sets/alphafold2
    14	INPUT_DIR=$SCRATCH/alphafold/fastas
    15	OUTPUT_DIR=$SCRATCH/alphafold/runs_cpu
    16	
    17	export SINGULARITY_CACHEDIR=$SCRATCH/singularity-cache
    18	export SINGULARITY_BIND="$INPUT_DIR,$OUTPUT_DIR,$DATABASE_DIR"
    19	
    20	export CONTAINER_IMAGE_DIR=/work/projects/singularity/ulhpc
    21	
    22	# To list input options type:
    23	# apptainer exec $CONTAINER_IMAGE_DIR/alphafold.sif python /app/alphafold/run_alphafold.py --help
    24	# If using GPUs then use the '--nv' flag, i.e. 'apptainer exec --nv ...'
    25	
    26	singularity exec $CONTAINER_IMAGE_DIR/alphafold-2.3.0.sif python /app/alphafold/run_alphafold.py \
    27	 --data_dir=$DATABASE_DIR \
    28	 --uniref90_database_path=$DATABASE_DIR/uniref90/uniref90.fasta \
    29	 --mgnify_database_path=$DATABASE_DIR/mgnify/mgy_clusters_2022_05.fa \
    30	 --bfd_database_path=$DATABASE_DIR/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    31	 --uniref30_database_path=$DATABASE_DIR/uniref30/UniRef30_2021_03 \
    32	 --pdb70_database_path=$DATABASE_DIR/pdb70/pdb70 \
    33	 --template_mmcif_dir=$DATABASE_DIR/pdb_mmcif/mmcif_files \
    34	 --obsolete_pdbs_path=$DATABASE_DIR/pdb_mmcif/obsolete.dat \
    35	 --model_preset=monomer \
    36	 --max_template_date=2022-1-1 \
    37	 --db_preset=full_dbs \
    38	 --output_dir=$OUTPUT_DIR \
    39	 --fasta_paths=$INPUT_DIR/test.fasta \
    40	 --use_gpu_relax=FALSE

We can see the standard slurm preamble from lines 1 to 8. Those parameters allow you to change the job name (to easily link the logs files to your experiments) or select the requested cpu count of memory mount.

Lines 11 to 20 prepares the environment and sets all the environment variables (database path, input and output files locations).

Lines 26 to 40 launch alphafold through the singularity image and specify all the parameters of the experiment.

### Update the launcher and prepare your environment

Note: you can use the nano text editor to update the launcher. Simply type:
	
	[clusteruser@access1 launchers]$ nano alphafold-cpu-launcher.sh
You can save with the ctrl-o shortcut and quit using the ctrl-x shortcut.

* Line 2 allows you to change the name of the job. The slurm log file (that contains the logs of the alphafold run, not the experiment results) will be named after this parameter; changing it for each experiment will allow you to keep track of what happened. **For this experiment, we will change the job name to _"alphafold-tutorial"_**
* Line 8 specifies the number of requested cpu cores. Beside adding more computation power, the side effect of this parameter is that is sets the amount of memory available for the computation (for reference, on aion, each requested cpu allocates around 2GB of RAM). **For this experiment, we will request 64 cores** (and thus 128GB of RAM), so change the value of _--cpus-per-task_ to 64.
* Line 14 sets the input directory that contain the fasta file for your experiment. Lets make it point to the tutorial fasta directory and change the value of __INPUT_DIR__ to **~/alphafold/fasta**
* Line 15 sets the output directory; i.e. where the results of the experiment will be stored
	* we first need to create the directory

	```[clusteruser@access1 launchers]$ mkdir ~/alphafold/output```
	
	* then we update _OUTPUT_DIR_ to point to ~/alphafold/output

### Launch the experiment

Submit your job to slurm using the sbatch command:

	[clusteruser@access1 launchers]$ sbatch alphafold-cpu-launcher.sh

You can see the status of your job using the sq command:

	[clusteruser@access1 launchers]$ 
	  JOBID PARTIT       QOS                 NAME        USER NODE  CPUS ST         TIME    TIME_LEFT PRIORITY NODELIST(REASON)
	2422240  batch    normal   alphafold-tutorial clusteruser    1    64 PD         0:00   2-00:00:00    10134 (Priority)

We can see that the name of the job is indeed _alphafold-tutorial_ and that we requested 64 CPUS. The status (_ST_ column) of the job is currently pending (_PD_), which means the job has not started yet. When rerunning the command after a couple of seconds, it should display running (_R_) in the _ST_ column.

When the experiment completes (it should take roughly 90 to 100 minutes), running this command againwill display an empty list.

While the experiment is running, you can have a look at the slurm log file. It should be named alphafold-tutorial-xxx.log where xxx corresponds to the job id from the sq command.

	[clusteruser@access1 launchers]$ tail -f alphafold-tutorial-2422240.log

Note 1: tail will display new lines as they are produced by the job

Note 2: you can quit quit tail with the ctrl-c shortcut

In the logs, you should quickly start to see some warnings:

	I0408 11:07:52.168599 139743818686976 xla_bridge.py:353] Unable to initialize backend 'cuda': FAILED_PRECONDITION: No visible GPU devices.
	I0408 11:07:52.168693 139743818686976 xla_bridge.py:353] Unable to initialize backend 'rocm': NOT_FOUND: Could not find registered platform with name: "rocm". Available platform names are: Host CUDA Interpreter
	I0408 11:07:52.169038 139743818686976 xla_bridge.py:353] Unable to initialize backend 'tpu': module 'jaxlib.xla_extension' has no attribute 'get_tpu_client'
	I0408 11:07:52.169125 139743818686976 xla_bridge.py:353] Unable to initialize backend 'plugin': xla_extension has no attributes named get_plugin_device_client. Compile TensorFlow with //tensorflow/compiler/xla/python:enable_plugin_device set to true (defaults to false) to enable this.
	W0408 11:07:52.169181 139743818686976 xla_bridge.py:360] No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)

This is expected as the aion cluster does not possess any accelerator. You can, on the last line, that alphafold default then on CPU.

You can then see the different steps of alphafold pipeline:

* Jackhammer uniref

```
	I0408 11:07:54.514855 139743818686976 utils.py:36] Started Jackhmmer (uniref90.fasta) query
	I0408 11:15:27.226291 139743818686976 utils.py:40] Finished Jackhmmer (uniref90.fasta) query in 452.711 seconds
```
* Jackhammer mgy

```
	I0408 11:15:27.239444 139743818686976 utils.py:36] Started Jackhmmer (mgy_clusters_2022_05.fa) query
	I0408 11:29:13.361756 139743818686976 utils.py:40] Finished Jackhmmer (mgy_clusters_2022_05.fa) query in 826.122 seconds
```
* HHblits

```
	I0408 11:29:33.086183 139743818686976 utils.py:36] Started HHblits query
	I0408 12:20:53.056995 139743818686976 utils.py:40] Finished HHblits query in 3079.971 seconds
```
* The actual prediction makes the rest of the logs.

We can notice that the database search takes most of the job runtime (around 72 minutes against the total run time of 100 minutes). Those steps are sequential, only uses 4 to 8 cpus and cannot be accelerated using gpus (limitation of alphafold).

### Looking at the results



## Running your experiments

In this section, we customize the launcher files to run other experiments.