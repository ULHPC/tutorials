[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/sequential/examples/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/sequential/examples/watermarking.md) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/sequential/examples/watermarking/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Serial tasks in action:  Watermarking images in Python

    Copyright (c) 2013-2019 UL HPC Team <hpc-team@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/sequential/examples/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/sequential/examples/slides.pdf)

The following github repositories will be used:

* [ULHPC/launcher-scripts](https://github.com/ULHPC/launcher-scripts)
    - **UPDATE (Dec 2020)** This repository is **deprecated** and kept for archiving purposes only. Consider the up-to-date launchers listed at the root of the ULHPC/tutorials repository, under `launchers/`
* [ULHPC/tutorials](https://github.com/ULHPC/tutorials)

----------
In this exercise, we will use another program, [`watermark.py`](scripts/watermark.py) (full path: `$SCRATCH/PS2/tutorials/sequential/examples/scripts/watermark.py`),
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


    (access IRIS)>$ si -N 1


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

    TASK="$SCRATCH/PS2/tutorials/sequential/examples/scripts/watermark.py"
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

--------------
## Conclusion

__At the end, please clean up your home and scratch directories :)__

**Please** do not store unnecessary files on the cluster's storage servers:

    (access)$> rm -rf $SCRATCH/PS2
