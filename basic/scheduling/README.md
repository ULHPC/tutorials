[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/beginners/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/beginners/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/beginners/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Job scheduling with SLURM

     Copyright (c) 2013-2020 UL HPC Team <hpc-sysadmins@uni.lu>

This page is part of the Getting started tutorial, and the follow-up of the "Overview" section.

* [reference documentation](https://hpc.uni.lu/users/docs/slurm.html)

## The basics

[Slurm](https://slurm.schedmd.com/) Slurm is an open source, fault-tolerant, and highly scalable cluster management and job scheduling system for large and small Linux clusters. It is used on Iris UL HPC cluster.

* It allocates exclusive or non-exclusive access to the resources (compute nodes) to users during a limited amount of time so that they can perform they work
* It provides a framework for starting, executing and monitoring work
* It arbitrates contention for resources by managing a queue of pending work.
* It permits to schedule jobs for users on the cluster resource

## Vocabulary

* A `user job` is characterized by:

    * the number of nodes
    * the number of CPU cores
    * the memory requested
    * the walltime
    * the launcher script, which will initiate your tasks


* Partition: group of compute nodes, with specific usage characteristics (time limits and maximum number of nodes per job

* QOS: The quality of service associated with a job affects the way it is scheduled (priority, preemption, limits per user, etc).

* Tasks: processes run in parallel inside the job

## Hands on

We will now see the basic commands of Slurm.

* Connect to **iris-cluster**. You can request resources in interactive mode:

        (access)$> si

  Notice that with no other parameters, srun gave you one resource for 30 minutes. You were also directly connected to the node you reserved with an interactive shell.
  Now exit the reservation (`exit` command or CTRL-D)

  When you run exit, you are disconnected and your reservation is terminated.

To avoid anticipated termination of your jobs in case of errors (terminal closed by mistake),
you can reserve and connect in two steps using the job id associated to your reservation.

* First run a passive job _i.e._ run a predefined command -- here `sleep 4h` to delay the execution for 4 hours -- on the first reserved node:

        (access)$> sbatch --qos normal --wrap "sleep 4h"
        Submitted batch job 390

  You noticed that you received a job ID (in the above example: `390`), which you can later use to connect to the reserved resource(s):

        (access)$> sjoin 390 # adapt the job ID accordingly ;)
        (node)$> ps aux | grep sleep
        <login> 186342  0.0  0.0 107896   604 ?        S    17:58   0:00 sleep 4h
        <login> 187197  0.0  0.0 112656   968 pts/0    S+   18:04   0:00 grep --color=auto sleep
        (node)$> exit             # or CTRL-D

**Question: At which moment the job `390` will end?**

a. after 10 days

b. after 2 hours

c. never, only when I'll delete the job

**Question: manipulate the `$SLURM_*` variables over the command-line to extract the following information, once connected to your job**

a. the list of hostnames where a core is reserved (one per line)
   * _hint_: `man echo`

b. number of reserved cores
   * _hint_: `search for the NPROCS variable`

c. number of reserved nodes
   * _hint_: `search for the NNODES variable`

d. number of cores reserved per node together with the node name (one per line)
   * Example of output:

            12 iris-11
            12 iris-15

   * _hint_: `NPROCS variable or NODELIST`


## Job management with interactive jobs 

Normally, the previously run job is still running.

* You can check the status of your running jobs using `squeue` command:

        (access)$> squeue             # list all jobs
        (access)$> squeue -u <login> # list all your jobs

  Then you can delete your job by running `scancel` command:

        (access)$> scancel 390


* You can see your system-level utilization (memory, I/O, energy) of a running job using `sstat $jobid`:

        (access)$> sstat 390

In all remaining examples of reservation in this section, remember to delete the reserved jobs afterwards (using `scancel` or `CTRL-D`)

You probably want to use more than one core, and you might want them for a different duration than one hour.

* Reserve interactively 4 cores in one task on one node, for 30 minutes (delete the job afterwards)

        (access)$> salloc -p interactive --time=0:30:0 -N 1 --ntasks-per-node=1 --cpus-per-task=4

* Reserve interactively 4 tasks (system processes) with 2 nodes for 30 minutes (delete the job afterwards)

        (access)$> salloc -p interactive --time=0:30:0 -N 2 --ntasks-per-node=4 --cpus-per-task=4

This command can also be written in a more compact way

        (access)$> si --time=0:30:0 -N2 -n4 -c2


* You can stop a waiting job from being scheduled and later, allow it to be scheduled:

        (access)$> scontrol hold $SLURM_JOB_ID
        (access)$> scontrol release $SLURM_JOB_ID

## Passive jobs submission

In the previous section, we have started interactive jobs using the `srun` command.
When `srun` is used on the access server, it can be used to submit an interactive jobs, meaning that the `srun` will wait for the job to be scheduled before starting a new shell.

The other possibility is to submit a passive job with `sbatch`.
In this case, resources can be specified on the command line as `srun`, or in the launcher script.

The launcher script first lines can contain headers, in the following case, in this case, the following launcher script will request a single task of one ore for 5 minutes, in the `batch` queue.

        #!/bin/bash -l
        #SBATCH -N 1
        #SBATCH --ntasks-per-node=1
        #SBATCH -c 1
        #SBATCH --time=0-00:05:00
        #SBATCH -p batch
        echo "Hello World!"

Save this content in a file named `launcher.sh` and submit it with `sbatch`

        (access)$> sbatch launcher.sh

Modify this example to:

* request 6 single core tasks equally spread across two nodes for 3 hours
* Request as many tasks as cores available on a single node in the batch queue for 3 hours
* Request one sequential task requiring half the memory of a regular iris node for 1 day (use the header `#SBATCH --mem=64GB`)
* Request one GPU tasks for 4 hours - dedicate 1/4 of available cores for its management (use the gpu partition and the header `#SBATCH -G 1`)
