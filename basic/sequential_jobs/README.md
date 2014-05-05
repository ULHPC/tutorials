HPC workflow with sequential jobs
=================================

# Prerequisites

Make sure you have followed the tutorial "Getting started".

# Intro

For many users, the typical usage of the HPC facilities is to execute 1 program with many parameters.
On your local machine, you can just start your program 100 times sequentially.
However, you will obtain better results if you are parallelize the executions on a HPC Cluster.

During this session, we will see 3 use cases:

* Exercise 1: Use the serial launcher (1 node, in sequential and parallel mode);
* Exercise 2: Use the generic launcher, distribute your executions on several nodes (python script);
* Exercise 3: Advanced use case, using a Java program: "JCell".

We will use the following github repositories:

* [ULHPC/launcher-scripts](https://github.com/ULHPC/launcher-scripts)
* [ULHPC/tutorial](https://github.com/ULHPC/tutorial)


# Practical session 2

## Connect the the cluster and set-up the environment for this tutorial

    ssh chaos-cluster

If your network connection is unstable, use [screen](http://www.mechanicalkeys.com/files/os/notes/tm.html):

    screen


We will use 2 directory:

* `$HOME`: default home directory, **backed up**, maximum 50GB, for important files
* `$WORK`: work directory, **non backed up**, maximum 500GB


Create a sub directory $WORK/PS2, and work inside it

    mkdir $WORK/PS2
    cd $WORK/PS2

In the following parts, we will assume that you are working in this directory.

Clone the repositories `ULHPC/tutorials` and `ULHPC/launcher-scripts.git`

    git clone https://github.com/ULHPC/launcher-scripts.git
    git clone https://github.com/ULHPC/tutorials.git




## Exercise 1: C program & circuit satisfiability

Get the source code and compile it on a node:

    oarsub -I
    cd $WORK/PS2
    gcc tutorials/basic/sequential_jobs/scripts/circuit.c -o circuit
    exit

We want to test all the parameters, so we create a parameter file with all the numbers from 1 to 65535.

    seq 65535 > $WORK/PS2/param_file


#### Step 1: Naive workflow

We will use the launcher `NAIVE_AKA_BAD_launcher_serial.sh` in `$WORK/PS2/launcher-scripts/bash/serial/`.

Edit the following variables:

* `TASK` must contain the path of the executable, 
* `ARG_TASK_FILE` must contain the path of your parameter file.

        TASK="$WORK/PS2/circuit"
        ARG_TASK_FILE=$WORK/PS2/param_file

Launch the job, in interactive mode:

    oarsub -I -l core=1

Execute the launcher:

    $WORK/PS2/launcher-scripts/bash/serial/NAIVE_AKA_BAD_launcher_serial.sh

Or in passive mode (the output will be written in a file named `OAR.<JOBID>.stdout`)

    oarsub -l core=1 $WORK/PS2/launcher-scripts/bash/serial/NAIVE_AKA_BAD_launcher_serial.sh



#### Step 2: Optimal method using GNU parallel (GNU Parallel)

We will use the launcher `launcher_serial.sh` in `$WORK/PS2/launcher-scripts/bash/serial/`.

Edit the following variables:

    TASK="$WORK/PS2/circuit"
    ARG_TASK_FILE=$WORK/PS2/param_file

Submit the job with `oarsub`

    oarsub -l nodes=1 $WORK/PS2/launcher-scripts/bash/serial/launcher_serial.sh


**Question**: compare and explain the execution time with both launchers:

* Naive workflow: time = 13m 30s
* Parallel workflow: time = 1m 23s

**/!\ In order to compare execution times, you must always use the same type of nodes (CPU/Memory), 
using [properties](https://hpc.uni.lu/users/docs/oar.html#select-nodes-precisely-with-properties)
in your `oarsub` command.**


## Exercise 2: Watermarking images in Python


We will use another program, `watermark.py`, and we will distribute the computation
on 2 nodes with the launcher `parallel_launcher.sh` in `$WORK/PS2/launcher-scripts/bash/generic``

This python script will apply a watermark to the images (using the Python Imaging library).

The command works like this:

    python watermark.py <path/to/watermark_image> <source_image>


We will work with 2 files:

* copyright.png: a transparent images, which can be applied as a watermark
* images.tgz: a compressed file, containing 30 JPG pictures (of the Gaia Cluster :) ).

#### Step 1: Prepare the input files

Copy the source files in your $WORK directory.

    tar xvf /tmp/images.tgz -C $WORK/PS2/
    cp /tmp/copyright.png $WORK/PS2

    cd $WORK/PS2

#### Step 2: Create a list of parameters

We must create a file containing a list of parameters, each line will be passed to `watermark.py`.

    ls -d -1 $WORK/PS2/images/*.JPG | awk -v watermark=$WORK/PS2/copyright.png '{print watermark " " $1}' > $WORK/PS2/generic_launcher_param
    \_____________________________/   \_________________________________________________________________/ \_________________________________/
                   1                                                    2                                                3

1. `ls -d -1`: list the images
2. `awk ...`: prefix each line with the first parameter (watermark file)
3. `>`: redirect the output to the file $WORK/generic_launcher_param


#### Step 3: Configure the launcher

We will use the launcher `parallel_launcher.sh` in `$WORK/PS2/launcher-scripts/bash/generic/`.

Edit the following variables:

    TASK="$WORK/PS2/tutorials/basic/sequential_jobs/scripts/watermark.py"
    ARG_TASK_FILE="$WORK/PS2/generic_launcher_param"
    # number of cores needed for 1 task
    NB_CORE_PER_TASK=2
    # Number of job slots
    NB_JOBS=12

#### Step 4: Submit the job

We will spawn 1 process / 2 cores

    oarsub -l nodes=2 $WORK/PS2/launcher-scripts/bash/generic/parallel_launcher.sh


#### Step 5: Download the files

On your laptop, transfer the files in the current directory and look at them with your favorite viewer:

    rsync -avz chaos-cluster:/work/users/<LOGIN>/PS2/images .


**Question**: which nodes are you using, identify your nodes with the command `oarstat -f -j <JOBID>` or Monika
([Chaos](https://hpc.uni.lu/chaos/monika), [Gaia](https://hpc.uni.lu/gaia/monika))


## Exercise 3: Advanced use case, using a Java program: "JCell"

Let's use [JCell](https://jcell.gforge.uni.lu/), a framework for working with genetic algorithms, programmed in Java.

We will use 3 scripts:

* `jcell_config_gen.sh`

We want to execute Jcell, and change the parameters MutationProb and CrossoverProb.
This script will install JCell, generate a tarball containing all the configuration files,
and the list of parameters to be given to the launcher.

* `jcell_wrapper.sh`

This script is a wrapper, and will start one execution of jcell with the configuration file given in parameter.
If a result already exists, then the execution will be skipped.
Thanks to this simple test, our workflow is fault tolerant, 
if the job is interrupted and restarted, only the missing results will be computed.

* `parallel_launcher.sh`

This script will drive the full experiment.


#### Step 1: Generate the configuration files:


        $WORK/PS2/tutorials/basic/sequential_jobs/scripts/jcell_config_gen.sh


  You will find the following files in `$WORK/PS2/jcell`:
  
  * `config.tgz`
  * `jcell_param`


#### Step 2: Edit the launcher configuration, in the file `$WORK/PS2/launcher-scripts/bash/generic/parallel_launcher.sh`.


        TASK="$WORK/PS2/tutorials/basic/sequential_jobs/scripts/jcell_wrapper.sh"
        ARG_TASK_FILE="$WORK/PS2/jcell/jcell_param"
        # number of cores needed for 1 task
        NB_CORE_PER_TASK=2
        # Number of job slots
        NB_JOBS=12


#### Step 3: Submit the job


        oarsub -l nodes=2 $WORK/PS2/launcher-scripts/bash/generic/parallel_launcher.sh


#### Step 4. Retrieve the results on your laptop:


        rsync -avz chaos-cluster:/work/users/<LOGIN>/PS2/jcell/results .


**Question**: check the system load and memory usage with Ganglia
([Chaos](https://hpc.uni.lu/chaos/ganglia), [Gaia](https://hpc.uni.lu/gaia/ganglia))


## At the end, please, clean up your home and work directories :)

Please, don't store unnecessary files on the cluster's storage servers

    rm -rf $WORK/PS2


# Going further:

* [Checkpoint / restart with BLCR](http://hpc.uni.lu/users/docs/oar.html#checkpointing)
* [OAR array jobs (fr)](http://crimson.oca.eu/spip.php?article157)

