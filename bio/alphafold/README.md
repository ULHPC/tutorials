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
	|-- ./fasta
	|	|-- ./fasta/test.fasta
	|-- ./launchers
		|-- ./launchers/alphafold-cpu-launcher.sh
		|-- ./launchers/alphafold-gpu-launcher.sh

- The fasta directory contains a sample sequence that we will use in the tutorial
- the launchers directory contain slurm launchers targeting either aion for CPU processing or iris for GPU processing.

### Check the Alphafold Singularity image

UL HPC provides Alphafold as a Singularity image. Those images are immutable and provide stable containerized  and independent environment each time they are started. As those images never change, they are great for reproducibility as well as versioning. At the time of writing, we propose the 2.3.0 Alphafold version. You can build other versions yourself or request the HLST to do it for you by opening a ticket on service now.

The images we build are automatically published on the cluster. You can see all the available images with the following command:

	[clusteruser@access1 alphafold]$ ls /work/projects/singularity/ulhpc/alphafold*
	alphafold-2.3.0.sif

You will have to update the launcher files to use other Alphafold versions.

### Check the the database files

The database files are stored inside a project that is publicly available. 
The models contained in the different versions of Alphafold have been trained to operate using specific version of databases. The 2.3.0 version of Alphafold provided on the cluster works with the databases provided in _/work/projects/bigdata_sets/alphafold2_. 

If you plan to use other versions of Alphafold, please be sure to update the launcher files to point to the relevant database files. If the files required by your version are not present, please open a ticket on service now.

## Step by step sample experiment

In this section, we show you how to prepare, start and review a small sample experiment.

## Running your experiments

In this section, we customize the launcher files to run other experiments.