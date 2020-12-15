[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/sequential/examples/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/sequential/examples/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/sequential/examples) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Serial tasks in action: Object recognition with Tensorflow and Python Imageai

    Copyright (c) 2013-2019 UL HPC Team <hpc-team@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/sequential/examples/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/sequential/examples/slides.pdf)

The following github repositories will be used:

* [ULHPC/launcher-scripts](https://github.com/ULHPC/launcher-scripts)
    - **UPDATE (Dec 2020)** This repository is **deprecated** and kept for archiving purposes only. Consider the up-to-date launchers listed at the root of the ULHPC/tutorials repository, under `launchers/`
* [ULHPC/tutorials](https://github.com/ULHPC/tutorials)

----------
In this exercise, we will process some images from the OpenImages V4 data set with an object recognition tools.

Create a file which contains the list of parameters (random list of images):

    (access)$>  find /work/projects/bigdata_sets/OpenImages_V4/train/ -print | head -n 10000 | sort -R | head -n 50 | tail -n +2 > $SCRATCH/PS2/param_file

## Step 0: Prepare the environment

    (access)$> srun -p interactive -N 1 --qos debug --pty bash -i

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


## Step 1: Naive workflow

We will use the launcher `NAIVE_AKA_BAD_launcher_serial.sh` (full path: `$SCRATCH/PS2/launcher-scripts/bash/serial/NAIVE_AKA_BAD_launcher_serial.sh`).

Edit the following variables:

* `MODULE_TO_LOAD` must contain the list of modules to load before executing `$TASK`,
* `TASK` must contain the path of the executable,
* `ARG_TASK_FILE` must contain the path of your parameter file.

        (node)$> nano $SCRATCH/PS2/launcher-scripts/bash/serial/NAIVE_AKA_BAD_launcher_serial.sh

            MODULE_TO_LOAD=(lang/Python)
            TASK="$SCRATCH/PS2/tutorials/sequential/examples/scripts/run_object_recognition.sh"
            ARG_TASK_FILE=$SCRATCH/PS2/param_file


### Using Slurm on Iris

Launch the job, in interactive mode and execute the launcher:

    (access)$> srun -p interactive -N 1 --qos debug --pty bash -i

    (node)$> source venv/bin/activate
    (node)$> $SCRATCH/PS2/launcher-scripts/bash/serial/NAIVE_AKA_BAD_launcher_serial.sh

**Or** in passive mode (the output will be written in a file named `BADSerial-<JOBID>.out`)

    (access)$> sbatch $SCRATCH/PS2/launcher-scripts/bash/serial/NAIVE_AKA_BAD_launcher_serial.sh


You can use the command `scontrol show job <JOBID>` to read all the details about your job:

    (access)$> scontrol show job 207001
    JobId=207001 JobName=BADSerial
       UserId=hcartiaux(5079) GroupId=clusterusers(666) MCS_label=N/A
       Priority=8791 Nice=0 Account=ulhpc QOS=normal
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

    (access)$> srun -p interactive --qos debug --jobid <JOBID> --pty bash

During the execution, you can see the job in the queue with the command `squeue`:

    (access)$> squeue
             JOBID PARTITION     NAME             USER ST       TIME  NODES NODELIST(REASON)
            207001     batch BADSeria        hcartiaux  R       2:27      1 iris-110


Using the [system monitoring tool ganglia](https://hpc.uni.lu/iris/ganglia/), check the activity on your node.


## Step 2: Optimal method using GNU parallel (GNU Parallel)

We will use the launcher `launcher_serial.sh` (full path: `$SCRATCH/PS2/launcher-scripts/bash/serial/launcher_serial.sh`).

Edit the following variables:

    (access)$> nano $SCRATCH/PS2/launcher-scripts/bash/serial/launcher_serial.sh

    MODULE_TO_LOAD=(lang/Python)
    TASK="$SCRATCH/PS2/tutorials/sequential/examples/scripts/run_object_recognition.sh"
    ARG_TASK_FILE=$SCRATCH/PS2/param_file

Submit the (passive) job with `sbatch`

    (access)$> sbatch $SCRATCH/PS2/launcher-scripts/bash/serial/launcher_serial.sh


**Question**: compare and explain the execution time with both launchers:


* Naive workflow: time = ?
  ![CPU usage for the sequential workflow](https://github.com/ULHPC/tutorials/raw/devel/sequential/examples/images/chaos_ganglia_seq.png)

* Parallel workflow: time = ?
  ![CPU usage for the parallel workflow](https://github.com/ULHPC/tutorials/raw/devel/sequential/examples/images/chaos_ganglia_parallel.png)

--------------
## Conclusion

__At the end, please clean up your home and scratch directories :)__

**Please** do not store unnecessary files on the cluster's storage servers:

    (access)$> rm -rf $SCRATCH/PS2
