## Application - Object recognition with Tensorflow and Python Imageai (an embarassingly parralel case)

### Introduction

For many users, the typical usage of the HPC facilities is to execute a single program with various parameters, 
which translates into executing sequentially a big number of independent tasks.

On your local machine, you can just start your program 1000 times sequentially.
However, you will obtain better results if you parallelize the executions on a HPC Cluster.

In this section, we will apply an object recognition script to random images from a dataset, first sequentially, then in parallel, and we will compare the execution time.

* [ULHPC/launcher-scripts](https://github.com/ULHPC/launcher-scripts)
* [ULHPC/tutorials](https://github.com/ULHPC/tutorials)


### Connect to the cluster access node, and set-up the environment for this tutorial

Create a sub directory $SCRATCH/PS2, and work inside it

```bash
(access)$> mkdir $SCRATCH/PS2
(access)$> cd $SCRATCH/PS2
```

At the end of the tutorial, please remember to remove this directory.

### Step 0: Prepare the environment

In the following parts, we will assume that you are working in this directory.

Clone the repositories `ULHPC/tutorials`

    (access)$> git clone https://github.com/ULHPC/tutorials.git


In this exercise, we will process some images from the OpenImages V4 data set with an object recognition tools.

Create a file which contains the list of parameters (random list of images):

    (access)$> find /work/projects/bigdata_sets/OpenImages_V4/train/ -print | head -n 10000 | sort -R | head -n 50 | tail -n +2 > $SCRATCH/PS2/param_file

Download a pre-trained model for image recognition

    (access)$> cd $SCRATCH/PS2
    (access)$> wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5

We will now prepare the software environment

    (access)$> srun -p interactive -N 1 --qos debug --pty bash -i

Load the default Python module

    (node) module load lang/Python

    (node) module list

Create a new python virtual env

    (node) cd $SCRATCH/PS2
    (node) virtualenv venv

Enable your newly created virtual env, and install the required modules inside.
This way, we will not pollute the home directory with the python modules installed in this exercise.

    (node) source venv/bin/activate

    (node) pip install tensorflow scipy opencv-python pillow matplotlib keras
    (node) pip install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.3/imageai-2.0.3-py3-none-any.whl

    (node) exit




### Step 1: Naive sequential workflow

We will create a new "launcher script", which is basically a loop over all the files listed 

        (node)$> nano $SCRATCH/PS2/launcher_sequential.sh

```bash
#!/bin/bash -l

#SBATCH --time=0-00:30:00 # 30 minutes
#SBATCH --partition=batch # Use the batch partition reserved for passive jobs
#SBATCH --qos=normal
#SBATCH -J BADSerial      # Set the job name
#SBATCH -N 1              # 1 computing node
#SBATCH -c 1              # 1 core

module load lang/Python

cd $SCRATCH/PS2
source venv/bin/activate

OUTPUT_DIR=$SCRATCH/PS2/object_recognition_$SLURM_JOBID
mkdir -p $OUTPUT_DIR

for SRC_FILE in $(cat $SCRATCH/PS2/param_file) ; do
    python $SCRATCH/PS2/tutorials/beginners/scripts/FirstDetection.py $SRC_FILE $OUTPUT_DIR
done
```


Submit the job in passive mode with `sbatch`

    (access)$> sbatch $SCRATCH/PS2/launcher_sequential.sh


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


### Step 2: Optimal method using GNU parallel (GNU Parallel)

We will create a new "launcher script", which uses GNU Parallel to execute 10 processes in parallel

        (node)$> nano $SCRATCH/PS2/launcher_parallel.sh

```bash
#!/bin/bash -l
#SBATCH --time=0-00:30:00 # 30 minutes
#SBATCH --partition=batch # Use the batch partition reserved for passive jobs
#SBATCH --qos=normal
#SBATCH -J ParallelExec   # Set the job name
#SBATCH -N 2              # 2 computing nodes
#SBATCH -n 10             # 10 tasks
#SBATCH -c 1              # 1 core per task

set -x
module load lang/Python

cd $SCRATCH/PS2
source venv/bin/activate

OUTPUT_DIR=$SCRATCH/PS2/object_recognition_$SLURM_JOBID
mkdir -p $OUTPUT_DIR

srun="srun --exclusive -N1 -n1"

parallel="parallel -j $SLURM_NTASKS --joblog runtask_$SLURM_JOBID.log --resume"

cat $SCRATCH/PS2/param_file | $parallel "$srun python $SCRATCH/PS2/tutorials/beginners/scripts/FirstDetection.py {} $OUTPUT_DIR"
```

Submit the job in passive mode with `sbatch`

    (access)$> sbatch $SCRATCH/PS2/launcher_parallel.sh


**Question**: compare and explain the execution time with both launchers:


* Naive workflow: time = ?
  ![CPU usage for the sequential workflow](https://github.com/ULHPC/tutorials/raw/devel/basic/sequential_jobs/images/chaos_ganglia_seq.png)

* Parallel workflow: time = ?
  ![CPU usage for the parallel workflow](https://github.com/ULHPC/tutorials/raw/devel/basic/sequential_jobs/images/chaos_ganglia_parallel.png)

**Bonus question**: transfer the generated files in `$SCRATCH/PS2/object_recognition_$SLURM_JOBID` to your laptop and visualize them


