Advanced HPC workflow with sequential jobs
==========================================

# Prerequisites

Make sure you have followed the tutorial "Getting started" and "HPC workflow with sequential jobs".

# Intro

During this session, we will review various advanced features of OAR:

* Exercise 1: Advanced OAR features: container, array jobs
* Exercise 2: Best effort jobs
* Exercise 3: Checkpoint restart

We will use the following github repositories:

* [ULHPC/launcher-scripts](https://github.com/ULHPC/launcher-scripts)
* [ULHPC/tutorials](https://github.com/ULHPC/tutorials)


# Container

## Connect the the cluster access node, and set-up the environment for this tutorial

    (yourmachine)$> ssh chaos-cluster

If your network connection is unstable, use [screen](http://www.mechanicalkeys.com/files/os/notes/tm.html):

    (access)$> screen


We will use 2 directory:

* `$HOME`: default home directory, **backed up**, maximum 50GB, for important files
* `$WORK`: work directory, **non backed up**, maximum 500GB


Create a sub directory $WORK/PS2, and work inside it

    (access)$> mkdir $WORK/PS6
    (access)$> cd $WORK/PS6

In the following parts, we will assume that you are working in this directory.

Clone the repositories `ULHPC/tutorials` and `ULHPC/launcher-scripts.git`

    (access)$> git clone https://github.com/ULHPC/launcher-scripts.git
    (access)$> git clone https://github.com/ULHPC/tutorials.git


## Exercise 1: useful oar features

### Container


With OAR, it is possible to execute jobs within another one.
This functionality is called **container jobs**.

It can be used to ensure you will have a pool of nodes in advance.
In example, if you want to create a container on the 29th of June, with 4 nodes, available during 10 hours

    oarsub -t container -r "2015-06-29 09:00:00" -l nodes=4,walltime=10:00:00

For testing purposes, you can create a small and short container with a passive job

    oarsub -t container -l nodes=2,walltime=0:30:00 "sleep 1800"
    [ADMISSION RULE] Modify resource description with type constraints
    OAR_JOB_ID=1428304

OAR gives you the job id of your container, you can submit jobs inside this container with the parameter `-t inner=<container id>`

    oarsub -I -t inner=1428304

Note that an inner job can not be a reservation (ie. it cannot overlap the container reservation).

### Array job

Let's create an array of 5 jobs. 
With an array, in one oarsub command, you can submit N sub jobs.

    oarsub --array 5 -n array_job_test -l /core=1 $WORK/PS6/tutorials/advanced/advanced_parametric_jobs/scripts/array_job.sh

    OAR_JOB_ID=1430652
    OAR_JOB_ID=1430653
    OAR_JOB_ID=1430654
    OAR_JOB_ID=1430655
    OAR_JOB_ID=1430656
    OAR_ARRAY_ID=1430652U

The command is started with the environment variable "OAR_ARRAY_INDEX".

Using this index, you can split your work load in N batches.

If you look at the output in the OAR stdout files, you should read something like that:

    cat OAR.array_job_test.*
    
    hostname :  s-cluster1-13
    OAR_JOB_NAME    : array_job_test
    OAR_JOB_ID      : 1430652
    OAR_ARRAY_ID    : 1430652
    OAR_ARRAY_INDEX : 1
    hostname :  s-cluster1-13
    OAR_JOB_NAME    : array_job_test
    OAR_JOB_ID      : 1430653
    OAR_ARRAY_ID    : 1430652
    OAR_ARRAY_INDEX : 2
    hostname :  s-cluster1-13
    OAR_JOB_NAME    : array_job_test
    OAR_JOB_ID      : 1430654
    OAR_ARRAY_ID    : 1430652
    OAR_ARRAY_INDEX : 3
    hostname :  s-cluster1-13
    OAR_JOB_NAME    : array_job_test
    OAR_JOB_ID      : 1430655
    OAR_ARRAY_ID    : 1430652
    OAR_ARRAY_INDEX : 4
    hostname :  s-cluster1-13
    OAR_JOB_NAME    : array_job_test
    OAR_JOB_ID      : 1430656
    OAR_ARRAY_ID    : 1430652
    OAR_ARRAY_INDEX : 5

Note: array jobs can neither be Interactive (-I) nor a reservation (-r).


## Exercise 2: Best effort job

By default, your jobs end in the default queue meaning they have all equivalent priority. 
You can also decide to create so called best-effort jobs which are scheduled in the besteffort queue.
Their particularity is that they are deleted if another not besteffort job wants resources where they are running.

The main advantage is that besteffort jobs allows you to bypass the limit on the default queue (number of jobs and walltime).

In order to submit a besteffort job, you must append the parameter "-t besteffort" to your oarsub command.
Here is an example:

    oarsub -t besteffort /path/to/prog

The probability of your best effort job being killed increases with the walltime of the job.
We strongly advise to send best effort job with small walltimes (in example 2 hours), 
and just resubmit your job if it is killed.

You can do that automatically with the `idempotent` type.
    
    oarsub -t besteffort -t idempotent /path/to/prog

If your idempotent job is killed by the OAR scheduler, then another job is automatically 
created and scheduled with same configuration.
Additionally, your job is also resubmitted if the exit code of your program is 99.

Our best effort launcher script implements this mechanism.

We will now submit a besteffort job using the script `launcher_besteffort.sh`.
This job will execute a R script computing 130 Fibonacci numbers, 
note that the script sleep 2 seconds in each iteration in order to simulate a long running job.

Edit the file `$WORK/PS6/launcher-scripts/bash/besteffort/launcher_besteffort.sh`

In the "Job settings" section, load R by adding this line:

    module load lang/R

Change the `$TASK` variable:

    TASK="Rscript $WORK/PS6/tutorials/advanced/advanced_parametric_jobs/scripts/test.R"


Now, submit the script.
    
    oarsub -t besteffort -t idempotent -l nodes=1 -p "network_address='h-cluster1-6'" $WORK/PS6/launcher-scripts/bash/besteffort/launcher_besteffort.sh

    oarstat -u

Once the script is started, submit a normal interactive job on the same node, in order to force OAR to kill the besteffort job

    oarsub -I -l nodes=1 -p "network_address='h-cluster1-6'"

    oarstat -u

Q: What do you observe ?


## Exercise 3: Checkpointing

**Checkpointing** is a technique which consists of storing a snapshot of the current application state,
in order to re-use it later for restarting the execution.
Checkpointing your jobs brings the following features:

* the job can overcome the default queue limit, especially the maximum walltime
* fault tolerance, your job can survive even if the node crash

You can implement a checkpointing mechanism yourself in your applications, or you can try with BLCR.
In all cases, your checkpointing mechanism must be chosen on a case by case basis.

Here, we will use BLCR, which works in some cases, but has some limitations, especially with sockets and network communication.

BLCR can be used with 3 commands:

* `cr_run` is a wrapper for your application
* `cr_checkpoint` will stop the application and save its state in a context file
* `cr_restart` will restart the application based on a context file

### Interactive example with Matlab

Create an interactive job, start a matlab program `cr_run`

    (access) oarsub -I

    (node) screen

    (node) module load base/MATLAB
    (node) export LIBCR_DISABLE_NSCD=YES
    (node) cr_run matlab -nojvm -nodisplay -nosplash <  $WORK/PS6/tutorials/advanced/advanced_parametric_jobs/scripts/test.m &

Get the process id of matlab

    (node) echo $!

Create a new window inside your screen with "Ctrl-a c".

Look at the process table on the node

    (node) ps aux | grep MATLAB

Stop the matlab process and register its state in the file `/tmp/test.context`

    (node) cr_checkpoint -f /tmp/test.context --kill -T <Matlab process id>

MATLAB should not be in the process table now

    (node) ps aux | grep MATLAB

Restart the process from the context file

    (node) cr_restart --no-restore-pid /tmp/test.context

The MATLAB process has been restarted

    (node) ps aux | grep MATLAB


### OAR integration

Checkpointing is supported by OAR, by combining several features:

* **besteffort** jobs: described in the previous exercise
* **idempotent** jobs: if your processus returns an exit code equal to 99, your job will be resubmitted with the same parameters ;
* **checkpoint parameter**: enable the checkpointing mechanism, specifies the time in seconds before sending a signal to the first processus of the job ;
* **signal parameter**: specify which signal to use when checkpointing (default is SIGUSR2).

Let's start a long Matlab program with the launcher `launcher_checkpoint_restart.sh` which uses all these features.

First in `$WORK/PS6/launcher-scripts/bash/besteffort/launcher_checkpoint_restart.sh`, modify the variable `TASK` with your program:

    TASK="matlab -nojvm -nodisplay -nosplash -r run('$WORK/PS6/tutorials/advanced/advanced_parametric_jobs/scripts/test.m');exit;"

In the "Job settings" section, load the Matlab module by adding this line:

    module load base/MATLAB

Submit your job with this command

    oarsub --checkpoint 30 --signal 12 -l walltime=00:02:00 -t besteffort -t idempotent $WORK/PS6/launcher-scripts/bash/besteffort/launcher_checkpoint_restart.sh

# Please, clean your files

    cd $WORK
    rm -rf PS6

