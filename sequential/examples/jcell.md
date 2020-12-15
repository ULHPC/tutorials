[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/sequential/examples/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/sequential/examples/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/sequential/examples) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Serial tasks in action: Genetic Algorithms Evolution with JCell

    Copyright (c) 2013-2019 UL HPC Team <hpc-team@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/sequential/examples/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/sequential/examples/slides.pdf)

The following github repositories will be used:

* [ULHPC/launcher-scripts](https://github.com/ULHPC/launcher-scripts)
    - **UPDATE (Dec 2020)** This repository is **deprecated** and kept for archiving purposes only. Consider the up-to-date launchers listed at the root of the ULHPC/tutorials repository, under `launchers/`
* [ULHPC/tutorials](https://github.com/ULHPC/tutorials)

----------
In this exercise, we will use [JCell](https://jcell.gforge.uni.lu/), a framework for working with genetic algorithms, programmed in Java.

We will use 3 scripts:

* [`jcell_config_gen.sh`](scripts/jcell_config_gen.sh) (full path: `$SCRATCH/PS2/tutorials/sequential/examples/scripts/jcell_config_gen.sh`)

We want to execute Jcell, and change the parameters MutationProb and CrossoverProb.
This script will install JCell, generate a tarball containing all the configuration files,
and the list of parameters to be given to the launcher.

* [`jcell_wrapper.sh`](scripts/jcell_wrapper.sh) (full path: `$SCRATCH/PS2/tutorials/sequential/examples/scripts/jcell_wrapper.sh`)

This script is a wrapper, and will start one execution of jcell with the configuration file given in parameter.
If a result already exists, then the execution will be skipped.
Thanks to this simple test, our workflow is fault tolerant,
if the job is interrupted and restarted, only the missing results will be computed.

* `parallel_launcher.sh` (full path: `$SCRATCH/PS2/launcher-scripts/bash/generic/parallel_launcher.sh`)

This script will drive the experiment, start and balance the java processes on all the reserved resources.


#### Step 1: Generate the configuration files:

Execute this script:

        (access)$> $SCRATCH/PS2/tutorials/sequential/examples/scripts/jcell_config_gen.sh


This script will generate the following files in `$SCRATCH/PS2/jcell`:

  * `config.tgz`
  * `jcell_param`


#### Step 2: Edit the launcher configuration, in the file `$SCRATCH/PS2/launcher-scripts/bash/generic/parallel_launcher.sh`.

This application is cpu-bound and not memory-bound, so we can set the value of `NB_CORE_PER_TASK` to 1.
Using these parameters, the launcher will spawn one java process per core on all the reserved nodes.

        (access)$> nano $SCRATCH/PS2/launcher-scripts/bash/generic/parallel_launcher.sh

        TASK="$SCRATCH/PS2/tutorials/sequential/examples/scripts/jcell_wrapper.sh"
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







--------------
## Conclusion

__At the end, please clean up your home and scratch directories :)__

**Please** do not store unnecessary files on the cluster's storage servers:

    (access)$> rm -rf $SCRATCH/PS2
