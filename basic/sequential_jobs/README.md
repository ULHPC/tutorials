[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/basic/sequential_jobs/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/basic/sequential_jobs/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/basic/sequential_jobs/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# HPC workflow with sequential jobs

     Copyright (c) 2013-2018 UL HPC Team <hpc-sysadmins@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/basic/sequential_jobs/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/basic/sequential_jobs/slides.pdf)

**Prerequisites**

Make sure you have followed the tutorial ["Getting started"](../getting_started/).

## Introduction

For many users, the typical usage of the HPC facilities is to execute 1 program with many parameters.
On your local machine, you can just start your program 100 times sequentially.
However, you will obtain better results if you parallelize the executions on a HPC Cluster.

During this session, we will see 3 use cases:

* _Exercise 1_: Use the serial launcher (1 node, in sequential and parallel mode);
* _Exercise 2_: Use the generic launcher, distribute your executions on several nodes (python script);
* _Exercise 3_: Advanced use case, using a Java program: "JCell".

We will use the following github repositories:

* [ULHPC/launcher-scripts](https://github.com/ULHPC/launcher-scripts)
* [ULHPC/tutorials](https://github.com/ULHPC/tutorials)

## Pre-requisites

### Connect to the cluster access node, and set-up the environment for this tutorial

You can chose one of the 3 production cluster hosted by the University of Luxembourg.

For the next sections, note that you will use `Slurm` on Iris.

```bash
(yourmachine)$> ssh iris-cluster
```

If your network connection is unstable, use [screen](http://www.mechanicalkeys.com/files/os/notes/tm.html):

```bash
(access)$> screen
```

We will work in [the home directory](https://hpc.uni.lu/users/docs/env.html#working-directories).

You can check the usage of your directories using the command `df-ulhpc`

```bash
(access)$> df-ulhpc
Directory                         Used  Soft quota  Hard quota  Grace period
---------                         ----  ----------  ----------  ------------
/home/users/hcartiaux             3.2G  100G        -           none
```

Note that the user directories are not yet all available on Iris, and that the quota are not yet enabled.

Create a sub directory $SCRATCH/PS2, and work inside it

```bash
(access)$> mkdir $SCRATCH/PS2
(access)$> cd $SCRATCH/PS2
```

In the following parts, we will assume that you are working in this directory.

Clone the repositories `ULHPC/tutorials` and `ULHPC/launcher-scripts.git`

    (access)$> git clone https://github.com/ULHPC/launcher-scripts.git
    (access)$> git clone https://github.com/ULHPC/tutorials.git

In order to edit files in your terminal, you are expected to use your preferred text editor:

* [nano](http://www.howtogeek.com/howto/42980/the-beginners-guide-to-nano-the-linux-command-line-text-editor/)
* [vim](http://vimdoc.sourceforge.net/htmldoc/usr_toc.html)
* [emacs](http://www.jesshamrick.com/2012/09/10/absolute-beginners-guide-to-emacs/)
* ...

If you have never used any of them, `nano` is intuitive, but `vim` and `emacs` are more powerful.

With nano, you will only have to learn a few shortcuts to get started:

* `$ nano <path/filename>`
* quit and save: `CTRL+x`
* save: `CTRL+o`
* highlight text: `Alt-a`
* Cut the highlighted text: `CTRL+k`
* Paste: `CTRL+u`


## Exercise 1: Object recognition with Tensorflow and Python Imageai

In this exercise, we will process some images from the OpenImages V4 data set with an object recognition tools.

Create a file which contains the list of parameters (random list of images):

    (access)$>  find /work/projects/bigdata_sets/OpenImages_V4/train/ -print | head -n 10000 | sort -R | head -n 50 | tail -n +2 > $SCRATCH/PS2/param_file

#### Step 0: Prepare the environment

    (access)$> srun -p interactive -N 1 --qos qos-interactive --pty bash -i

Load the default Python module

    (node) module load lang/Python

    (node) module list

Create a new python virtual env

    (node) cd $SCRATCH/PS2
    (node) virtualenv venv

Enable your newly created virtual env, and install the required modules inside

    (node) source venv/bin/activate

    (node) pip install tensorflow scipy opencv-python pillow matplotlib keras
    (node) pip install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl

    (node) exit


#### Step 1: Naive workflow

We will use the launcher `NAIVE_AKA_BAD_launcher_serial.sh` (full path: `$SCRATCH/PS2/launcher-scripts/bash/serial/NAIVE_AKA_BAD_launcher_serial.sh`).

Edit the following variables:

* `MODULE_TO_LOAD` must contain the list of modules to load before executing `$TASK`,
* `TASK` must contain the path of the executable,
* `ARG_TASK_FILE` must contain the path of your parameter file.

        (node)$> nano $SCRATCH/PS2/launcher-scripts/bash/serial/NAIVE_AKA_BAD_launcher_serial.sh

            MODULE_TO_LOAD=(lang/Python)
            TASK="$SCRATCH/PS2/tutorials/basic/sequential_jobs/scripts/run_object_recognition.sh"
            ARG_TASK_FILE=$SCRATCH/PS2/param_file


##### Using Slurm on Iris

Launch the job, in interactive mode and execute the launcher:

    (access)$> srun -p interactive -N 1 --qos qos-interactive --pty bash -i

    (node)$> source venv/bin/activate
    (node)$> $SCRATCH/PS2/launcher-scripts/bash/serial/NAIVE_AKA_BAD_launcher_serial.sh

**Or** in passive mode (the output will be written in a file named `BADSerial-<JOBID>.out`)

    (access)$> sbatch $SCRATCH/PS2/launcher-scripts/bash/serial/NAIVE_AKA_BAD_launcher_serial.sh


You can use the command `scontrol show job <JOBID>` to read all the details about your job:

    (access)$> scontrol show job 207001
    JobId=207001 JobName=BADSerial
       UserId=hcartiaux(5079) GroupId=clusterusers(666) MCS_label=N/A
       Priority=8791 Nice=0 Account=ulhpc QOS=qos-batch
       JobState=RUNNING Reason=None Dependency=(null)
       Requeue=0 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0
       RunTime=00:00:23 TimeLimit=01:00:00 TimeMin=N/A
       SubmitTime=2018-11-23T10:01:04 EligibleTime=2018-11-23T10:01:04
       StartTime=2018-11-23T10:01:05 EndTime=2018-11-23T11:01:05 Deadline=N/A


And the command `sacct` to see the start and end date


    (access)$> sacct --format=start,end --j 207004
                  Start                 End
    ------------------- -------------------
    2018-11-23T10:01:20 2018-11-23T10:02:31
    2018-11-23T10:01:20 2018-11-23T10:02:31

In all cases, you can connect to a reserved node using the command `srun`
and check the status of the system using standard linux command (`free`, `top`, `htop`, etc)

    (access)$> srun -p interactive --qos qos-interactive --jobid <JOBID> --pty bash

During the execution, you can see the job in the queue with the command `squeue`:

    (access)$> squeue
             JOBID PARTITION     NAME             USER ST       TIME  NODES NODELIST(REASON)
            207001     batch BADSeria        hcartiaux  R       2:27      1 iris-110


Using the [system monitoring tool ganglia](https://hpc.uni.lu/iris/ganglia/), check the activity on your node.


#### Step 2: Optimal method using GNU parallel (GNU Parallel)

We will use the launcher `launcher_serial.sh` (full path: `$SCRATCH/PS2/launcher-scripts/bash/serial/launcher_serial.sh`).

Edit the following variables:

    (access)$> nano $SCRATCH/PS2/launcher-scripts/bash/serial/launcher_serial.sh

    MODULE_TO_LOAD=(lang/Python)
    TASK="$SCRATCH/PS2/tutorials/basic/sequential_jobs/scripts/run_object_recognition.sh"
    ARG_TASK_FILE=$SCRATCH/PS2/param_file

Submit the (passive) job with `sbatch`

    (access)$> sbatch $SCRATCH/PS2/launcher-scripts/bash/serial/launcher_serial.sh


**Question**: compare and explain the execution time with both launchers:


* Naive workflow: time = ?
  ![CPU usage for the sequential workflow](https://github.com/ULHPC/tutorials/raw/devel/basic/sequential_jobs/images/chaos_ganglia_seq.png)

* Parallel workflow: time = ?
  ![CPU usage for the parallel workflow](https://github.com/ULHPC/tutorials/raw/devel/basic/sequential_jobs/images/chaos_ganglia_parallel.png)




## Exercise 2: Watermarking images in Python


We will use another program, `watermark.py` (full path: `$SCRATCH/PS2/tutorials/basic/sequential_jobs/scripts/watermark.py`),
and we will distribute the computation on 2 nodes with the launcher `parallel_launcher.sh`
(full path: `$SCRATCH/PS2/launcher-scripts/bash/generic/parallel_launcher.sh`).

This python script will apply a watermark to the images (using the Python Imaging library).

The command works like this:

    python watermark.py <path/to/watermark_image> <source_image>


We will work with 2 files:

* `copyright.png`: a transparent images, which can be applied as a watermark
* `images.tgz`: a compressed file, containing 30 JPG pictures (of the Gaia Cluster :) ).

#### Step 0: python image manipulation module installation

In an interactive job, install `pillow` in your home directory using this command:


    (access IRIS)>$ srun -p interactive -N 1 --qos qos-interactive --pty bash -i


    (node)>$ pip install --user pillow

#### Step 1: Prepare the input files

Copy the source files in your $SCRATCH directory.

    (access)>$ tar xvf /mnt/isilon/projects/ulhpc-tutorials/sequential/images2.tgz -C $SCRATCH/PS2/
    (access)>$ cp /mnt/isilon/projects/ulhpc-tutorials/sequential/ulhpc_logo.png $SCRATCH/PS2

    (access)>$ cd $SCRATCH/PS2

#### Step 2: Create a list of parameters

We must create a file containing a list of parameters, each line will be passed to `watermark.py`.

    ls -d -1 $SCRATCH/PS2/images/*.JPG | awk -v watermark=$SCRATCH/PS2/ulhpc_logo.png '{print watermark " " $1}' > $SCRATCH/PS2/generic_launcher_param
    \_____________________________/   \_________________________________________________________________/ \_________________________________/
                   1                                                    2                                                3

1. `ls -d -1`: list the images
2. `awk ...`: prefix each line with the first parameter (watermark file)
3. `>`: redirect the output to the file $SCRATCH/generic_launcher_param


#### Step 3: Configure the launcher

We will use the launcher `parallel_launcher.sh` (full path: `$SCRATCH/PS2/launcher-scripts/bash/generic/parallel_launcher.sh`).

Edit the following variables:


    (access)$> nano $SCRATCH/PS2/launcher-scripts/bash/generic/parallel_launcher.sh

    TASK="$SCRATCH/PS2/tutorials/basic/sequential_jobs/scripts/watermark.py"
    ARG_TASK_FILE="$SCRATCH/PS2/generic_launcher_param"
    # number of cores needed for 1 task
    NB_CORE_PER_TASK=2

#### Step 4: Submit the job

We will spawn 1 process per 2 cores on 2 nodes

On Iris, the Slurm job submission command is `sbatch`

    (access IRIS)>$ sbatch $SCRATCH/PS2/launcher-scripts/bash/generic/parallel_launcher.sh

#### Step 5: Download the files

On your laptop, transfer the files in the current directory and look at them with your favorite viewer.
Use one of these commands according to the cluster you have used:

    (yourmachine)$> rsync -avz iris-cluster:/scratch/users/<LOGIN>/PS2/images .


**Question**: which nodes are you using, identify your nodes with the command `sacct` or Slurmweb 


## Exercise 3: Advanced use case, using a Java program: "JCell"

Let's use [JCell](https://jcell.gforge.uni.lu/), a framework for working with genetic algorithms, programmed in Java.

We will use 3 scripts:

* `jcell_config_gen.sh` (full path: `$SCRATCH/PS2/tutorials/basic/sequential_jobs/scripts/jcell_config_gen.sh`)

We want to execute Jcell, and change the parameters MutationProb and CrossoverProb.
This script will install JCell, generate a tarball containing all the configuration files,
and the list of parameters to be given to the launcher.

* `jcell_wrapper.sh` (full path: `$SCRATCH/PS2/tutorials/basic/sequential_jobs/scripts/jcell_wrapper.sh`)

This script is a wrapper, and will start one execution of jcell with the configuration file given in parameter.
If a result already exists, then the execution will be skipped.
Thanks to this simple test, our workflow is fault tolerant,
if the job is interrupted and restarted, only the missing results will be computed.

* `parallel_launcher.sh` (full path: `$SCRATCH/PS2/launcher-scripts/bash/generic/parallel_launcher.sh`)

This script will drive the experiment, start and balance the java processes on all the reserved resources.


#### Step 1: Generate the configuration files:

Execute this script:

        (access)$> $SCRATCH/PS2/tutorials/basic/sequential_jobs/scripts/jcell_config_gen.sh


This script will generate the following files in `$SCRATCH/PS2/jcell`:

  * `config.tgz`
  * `jcell_param`


#### Step 2: Edit the launcher configuration, in the file `$SCRATCH/PS2/launcher-scripts/bash/generic/parallel_launcher.sh`.

This application is cpu-bound and not memory-bound, so we can set the value of `NB_CORE_PER_TASK` to 1.
Using these parameters, the launcher will spawn one java process per core on all the reserved nodes.

        (access)$> nano $SCRATCH/PS2/launcher-scripts/bash/generic/parallel_launcher.sh

        TASK="$SCRATCH/PS2/tutorials/basic/sequential_jobs/scripts/jcell_wrapper.sh"
        ARG_TASK_FILE="$SCRATCH/PS2/jcell/jcell_param"
        # number of cores needed for 1 task
        NB_CORE_PER_TASK=1

#### Step 3: Submit the job


On Iris, the Slurm job submission command is `sbatch`

    (access IRIS)>$ sbatch $SCRATCH/PS2/launcher-scripts/bash/generic/parallel_launcher.sh


#### Step 4. Retrieve the results on your laptop:

Use one of these commands according to the cluster you have used:

        (yourmachine)$> rsync -avz iris-cluster:/scratch/users/<LOGIN>/PS2/jcell/results .


**Question**: check the system load and memory usage with [Ganglia](https://hpc.uni.lu/iris/ganglia)


## Conclusion

__At the end, please clean up your home and scratch directories :)__

**Please** do not store unnecessary files on the cluster's storage servers:

    (access)$> rm -rf $SCRATCH/PS2


For going further:

* [Checkpoint / restart with BLCR](http://hpc.uni.lu/users/docs/oar.html#checkpointing)
* [OAR array jobs (fr)](http://crimson.oca.eu/spip.php?article157)
