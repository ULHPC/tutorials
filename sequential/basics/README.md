[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/basic/sequential_jobs/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/basic/sequential_jobs/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/basic/sequential_jobs/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# HPC workflow with sequential jobs

     Copyright (c) 2013-2019 UL HPC Team <hpc-sysadmins@uni.lu>

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
* _Exercise 4_: Advanced use case, distributing embarrassingly parallel tasks with GNU Parallel within a slurm job

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


--------------------------------------------------------------------
## Exercise 1: Object recognition with Tensorflow and Python Imageai

In this exercise, we will process some images from the OpenImages V4 data set with an object recognition tools.

Create a file which contains the list of parameters (random list of images):

    (access)$>  find /work/projects/bigdata_sets/OpenImages_V4/train/ -print | head -n 10000 | sort -R | head -n 50 | tail -n +2 > $SCRATCH/PS2/param_file

#### Step 0: Prepare the environment

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




---------------------------------------------------------------
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


    (access IRIS)>$ srun -p interactive -N 1 --qos debug --pty bash -i


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


---------------------------------------------------------------
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


---------------------------------------------------------------
## Exercise 4: Advanced use case, distributing embarrassingly parallel tasks with GNU Parallel

[GNU Parallel](http://www.gnu.org/software/parallel/)) is a tool for executing tasks in parallel, typically on a single machine. When coupled with the Slurm command `srun`, parallel becomes a powerful way of distributing a set of tasks amongst a number of workers. This is particularly useful when the number of tasks is significantly larger than the number of available workers (i.e. `$SLURM_NTASKS`), and each tasks is independent of the others.

To illustrate the advantages of this approach, a sample launcher script is proposed under **[`scripts/launcher.parallel.sh`](https://github.com/ULHPC/tutorials/blob/devel/basic/sequential_jobs/scripts/launcher.parallel.sh)**.

It will invoke the command [`stress`](https://linux.die.net/man/1/stress) to impose a CPU load during 60s 8 times (with an increasing hang time, i.e. 1 to 8s).
__We have thus 8 tasks, and we will create a single job handling this execution__

More precisely, each task consists of the following command:

``` bash
stress --cpu 1 --timeout 60s --vm-hang <n>
```

* If run __sequentially__, this workflow would take at least 8x60 = 480s i.e. __8 min__
* we will invoke the proposed launcher which will bundle the execution using __NO MORE THAN__ `$SLURM_NTASKS` (4 in the below tests), i.e. in approximately 2 minutes

``` bash
# Go to the appropriate directory
$> cd  $SCRATCH/PS2/tutorials/basic/sequential_jobs
$> ./scripts/launcher.parallel.sh -h
NAME
    launcher.parallel.sh [-n] [TASK]

    Using GNU parallel within a single node to run embarrasingly parallel
    problems, i.e. execute multiple times the command '${TASK}' within a
    'tunnel' set to run NO MORE THAN ${SLURM_NTASKS} tasks in parallel.

    State of the execution is stored in logs/state.parallel.log and is used to
    resume the execution later on, from where it stoppped (either due to the
    fact that the slurm job has been stopped by failure or by hitting a walltime
    limit) next time you invoke this script.
    In particular, if you need to rerun this GNU Parallel job, be sure to delete
    the logfile logs/state*.parallel.log or it will think it has already
    finished!

    By default, the 'stress --cpu 1 --timeout 60s --vm-hang <arg>' command is executed
    with the arguments {1..8}

OPTIONS
  -n --noop --dry-run:   dry run mode

EXAMPLES
  Within an interactive job (use --exclusive for some reason in that case)
      (access)$> si --exclusive --ntasks-per-node 4
      (node)$> ./scripts/launcher.parallel.sh -n    # dry-run
      (node)$> ./scripts/launcher.parallel.sh
  Within a passive job
      (access)$> sbatch --ntasks-per-node 4 ./scripts/launcher.parallel.sh
  Within a passive job, using several cores (6) per tasks
      (access)$> sbatch --ntasks-per-socket 2 --ntasks-per-node 4 -c 6 ./scripts/launcher.parallel.sh

  Get the most interesting usage statistics of your jobs <JOBID> (in particular
  for each job step) with:
     sacct -j <JOBID> --format User,JobID,Jobname,partition,state,time,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist,ConsumedEnergyRaw
```

#### Step 1: dry-run tests in an interactive jobs

For some reason, if you intend to really execute this script within the `interactive` partition, you will need to use the `--exclusive` flag.
By default, we will only make a dry-run tests in this case

``` bash
# Get an interactive job -- use '--exclusive' (but everybody cannot be serve in
# this case) if you really intend to run the commands
(access)$> srun -p interactive --exclusive --ntasks-per-node 4 --pty bash
# OR, simplier: 'si --exclusive --ntasks-per-node 4'
(node)$> echo $SLURM_NTASKS
4
(node)$> ./scripts/launcher.parallel.sh -n
### Starting timestamp (s): 1576074072
parallel --delay .2 -j 4 --joblog logs/state.parallel.log --resume srun  --exclusive -n1 -c 1 --cpu-bind=cores stress --cpu 1 --timeout 60s --vm-hang {1} ::: 1 2 3 4 5 6 7 8
### Ending timestamp (s): 1576074072"
# Elapsed time (s): 0

Beware that the GNU parallel option --resume makes it read the log file set by
--joblog (i.e. logs/state*.log) to figure out the last unfinished task (due to the
fact that the slurm job has been stopped due to failure or by hitting a walltime
limit) and continue from there.
In particular, if you need to rerun this GNU Parallel job, be sure to delete the
logfile logs/state*.parallel.log or it will think it has already finished!

# Release your job
(node)$> exit    # OR 'CTRL+D'
```

#### Step 2: real test in passive job

Submit this job using `sbatch`

``` bash
(access)$> sbatch ./scripts/launcher.parallel.sh
```

Check that you have a single job in the queue, assigned as many nodes/cores as
was requested:

``` bash
(access)$> sq    # OR 'squeue -u $(whoami)'  OR 'sqs'
             JOBID PARTITION                           NAME     USER ST       TIME  TIME_LEFT  NODES NODELIST(REASON)
           1359477     batch                    GnuParallel svarrett  R       0:00    1:00:00      1 iris-093
```

In this launcher script, GNU Parallel maintains a log of the work that has already been done (under `logs/state.parallel.log`), along with the exit value of each step (useful for determining any failed steps).

``` bash
(access)$> tail logs/state.parallel.log
Seq     Host    Starttime       JobRuntime      Send    Receive Exitval Signal  Command
1       :       1576074282.818      60.116      0       121     0       0       srun  --exclusive -n1 -c 1 --cpu-bind=cores stress --cpu 1 --timeout 60s --vm-hang 1
2       :       1576074283.041      60.127      0       121     0       0       srun  --exclusive -n1 -c 1 --cpu-bind=cores stress --cpu 1 --timeout 60s --vm-hang 2
3       :       1576074283.244      60.134      0       121     0       0       srun  --exclusive -n1 -c 1 --cpu-bind=cores stress --cpu 1 --timeout 60s --vm-hang 3
4       :       1576074283.457      60.128      0       121     0       0       srun  --exclusive -n1 -c 1 --cpu-bind=cores stress --cpu 1 --timeout 60s --vm-hang 4
```

As indicated in the help message, the state of the execution is stored in the joblog file `logs/state.parallel.log` which is used to resume the execution later on from `--resume`, in case your job is stopped, either due to the fact that the slurm job has been stopped (by failure) or by hitting the walltime limit).
To resume from a past execution, you simply need to re-run the script.

**`/!\ IMPORTANT`** In particular, if you need to rerun this GNU Parallel job, be sure to delete the joblog file `logs/state.parallel.log` or it will think it has already finished!

``` bash
rm logs/state.parallel.log
```

You can notice the slurm logfile set to `GnuParallel-<JOBID>.out`

``` bash
(access)$> cat GnuParallel-*.out

```

#### Step 3: get the usage statistics of your job

You can extract from the slurm database the usage statistics of this job, in particilar with regards the CPU and energy consumption for each job step corresponding to a GNU parallel task.

``` bash
# /!\ ADAPT <JOBID> with the appropriate Job ID. Ex: 1359477
$> sacct -j 1359477 --format User,JobID,Jobname,partition,state,time,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist,ConsumedEnergyRaw
     User        JobID    JobName  Partition      State  Timelimit    Elapsed     MaxRSS  MaxVMSize   NNodes      NCPUS        NodeList ConsumedEnergyRaw
--------- ------------ ---------- ---------- ---------- ---------- ---------- ---------- ---------- -------- ---------- --------------- -----------------
svarrette 1359477      GnuParall+      batch  COMPLETED   01:00:00   00:02:01                              1          4        iris-093             26766
          1359477.bat+      batch             COMPLETED              00:02:01     23948K    178784K        1          4        iris-093             26747
          1359477.ext+     extern             COMPLETED              00:02:01          0    107956K        1          4        iris-093             26766
          1359477.0        stress             COMPLETED              00:01:00       124K    248536K        1          1        iris-093             13380
          1359477.1        stress             COMPLETED              00:01:01       124K    248536K        1          1        iris-093             13391
          1359477.2        stress             COMPLETED              00:01:00       124K    248536K        1          1        iris-093             13402
          1359477.3        stress             COMPLETED              00:01:00       124K    248536K        1          1        iris-093             13408
          1359477.4        stress             COMPLETED              00:01:00       124K    248536K        1          1        iris-093             13121
          1359477.5        stress             COMPLETED              00:01:00       124K    248536K        1          1        iris-093             13118
          1359477.6        stress             COMPLETED              00:01:00       124K    248536K        1          1        iris-093             13117
          1359477.7        stress             COMPLETED              00:01:00       124K    248536K        1          1        iris-093             13118

# Another nice slurm utility is 'seff <JOBID>'
$> seff 1359477
Job ID: 1359477
Cluster: iris
User/Group: svarrette/clusterusers
State: COMPLETED (exit code 0)
Nodes: 1
Cores per node: 4
CPU Utilized: 00:08:00
CPU Efficiency: 99.17% of 00:08:04 core-walltime
Job Wall-clock time: 00:02:01
Memory Utilized: 23.39 MB
Memory Efficiency: 0.14% of 16.00 GB
```

As can be seen, the jobs was using quite efficiently the allocated CPUs, but not the memory.


--------------
## Conclusion

__At the end, please clean up your home and scratch directories :)__

**Please** do not store unnecessary files on the cluster's storage servers:

    (access)$> rm -rf $SCRATCH/PS2


For going further:

* [Distributing Tasks with SLURM and GNU Parallel](https://www.marcc.jhu.edu/getting-started/additional-resources/distributing-tasks-with-slurm-and-gnu-parallel/)
* [Automating large numbers of tasks](https://rcc.uchicago.edu/docs/tutorials/kicp-tutorials/running-jobs.html)
* (old) [Checkpoint / restart with BLCR](http://hpc.uni.lu/users/docs/oar.html#checkpointing)
* (old) [OAR array jobs (fr)](http://crimson.oca.eu/spip.php?article157)
