[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/bio/basics/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/bio/basics/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/bio/basics/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Bioinformatics software on the UL HPC platform

Copyright (c) 2014-2018 UL HPC Team  <hpc-sysadmins@uni.lu>

Authors: Valentin Plugaru and Sarah Peter

[![](https://github.com/ULHPC/tutorials/raw/devel/bio/basics/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/bio/basics/slides.pdf)

The objective of this tutorial is to exemplify the execution of several Bioinformatics packages on top of the [UL HPC](http://hpc.uni.lu) platform.

The targeted applications are:

* [ABySS](http://www.bcgsc.ca/platform/bioinfo/software/abyss)
* [Gromacs](http://www.gromacs.org/)
* [Bowtie2](http://bowtie-bio.sourceforge.net/bowtie2/index.shtml) / [TopHat](http://ccb.jhu.edu/software/tophat/index.shtml)
* [mpiBLAST](http://www.mpiblast.org/)

The tutorial will:

1. show you how to load and run pre-configured versions of these applications on the clusters
2. show you how to download and use updated versions of Bowtie2/TopHat
3. discuss the parallelization capabilities of these applications

## Prerequisites

When you look at the [software page](https://hpc.uni.lu/users/software/) you will notice that some of the applications are part of the *bioinfo* software set. The modules in this set are not visible by default. To use them within a job you have to do:

	(node)$> module use /opt/apps/resif/data/stable/bioinfo/modules/all

If you want them to always be available, you can add the following line to your `.bash_private`:

	command -v module >/dev/null 2>&1 && module use /opt/apps/resif/data/stable/bioinfo/modules/all

This tutorial relies on several input files for the bioinformatics packages, thus you will need to download them before following the instructions in the next sections:

    (access)$> mkdir -p ~/bioinfo-tutorial/gromacs ~/bioinfo-tutorial/tophat ~/bioinfo-tutorial/mpiblast
    (access)$> cd ~/bioinfo-tutorial
    (access)$> wget --no-check-certificate https://raw.github.com/ULHPC/tutorials/devel/bio/basics/gromacs/pr.tpr -O gromacs/pr.tpr
    (access)$> wget --no-check-certificate https://raw.github.com/ULHPC/tutorials/devel/bio/basics/tophat/test_data.tar.gz -O tophat/test_data.tar.gz
    (access)$> wget --no-check-certificate https://raw.github.com/ULHPC/tutorials/devel/bio/basics/tophat/test2_path -O tophat/test2_path
    (access)$> wget --no-check-certificate https://raw.github.com/ULHPC/tutorials/devel/bio/basics/mpiblast/test.fa -O mpiblast/test.fa

Or simply clone the full tutorials repository and make a link to the Bioinformatics tutorial:

    (access)$> git clone https://github.com/ULHPC/tutorials.git
    (access)$> ln -s tutorials/advanced/Bioinformatics/ ~/bioinfo-tutorial


## ABySS
Characterization: CPU intensive, data intensive, native parallelization

### Description

__ABySS__: Assembly By Short Sequences

ABySS is a de novo, parallel, paired-end sequence assembler that is designed for short reads.
The single-processor version is useful for assembling genomes up to 100 Mbases in size.
The parallel version is implemented using MPI and is capable of assembling larger genomes [\[\*\]](http://www.bcgsc.ca/platform/bioinfo/software/abyss).

### Example

This example will be ran in an interactive session, with batch-mode executions
being proposed later on as exercises.

* Gaia

		# Connect to Gaia (Linux/OS X):
		(yourmachine)$> ssh access-gaia.uni.lu
		
		# Request 1 full node in an interactive job:
		(access-gaia)$> oarsub -I -l nodes=1,walltime=00:30:00

* Iris

		# Connect to Iris (Linux/OS X):
		(yourmachine)$> ssh access-iris.uni.lu
		
		# Request half a node in an interactive job:
		(access-iris)$> srun -p interactive --qos qos-interactive -t 0-0:30:0 -N 1 -c 1 --ntasks-per-node=14 --pty bash


```
# Load bioinfo software set
(node)$> module use /opt/apps/resif/data/stable/bioinfo/modules/all

# Check the ABySS versions installed on the clusters:
(node)$> module avail 2>&1 | grep -i abyss

# Load the default ABySS version:
(node)$> module load bio/ABySS

# Check that it has been loaded, along with its dependencies:
(node)$> module list

# All the ABySS binaries are now in your path (check with TAB autocompletion)
(node)$> abyss-<TAB>
```

In the ABySS package only the `ABYSS-P` application is parallelized using MPI and can be run on several cores (and across several nodes) using
the `abyss-pe` launcher.

    # Create a test directory and go to it
    (node)$> mkdir ~/bioinfo-tutorial/abyss
    (node)$> cd ~/bioinfo-tutorial/abyss
    
    # Set the input files' directory in the environment
    (node)$> export ABYSSINPDIR=/mnt/isilon/projects/ulhpc-tutorials/bioinformatics/abyss
    
    # Give a name to the experiment
    (node)$> export ABYSSNAME='abysstest'

* Gaia

		# Set the number of cores to use based on OAR's host file
		(node)$> export ABYSSNPROC=$(cat $OAR_NODEFILE | wc -l)
		
		# Launch the paired end assembler:
		(node)$> abyss-pe mpirun="mpirun -x PATH -x LD_LIBRARY_PATH -hostfile $OAR_NODEFILE" name=${ABYSSNAME} np=${ABYSSNPROC} k=31 n=10 lib=pairedend pairedend="${ABYSSINPDIR}/SRR001666_1.fastq.bz2 ${ABYSSINPDIR}/SRR001666_2.fastq.bz2" > ${ABYSSNAME}.out 2> ${ABYSSNAME}.err

* Iris

		# Set the number of cores to use based on SLURM environment variables
		(node)$> export ABYSSNPROC=$(expr $SLURM_NNODES \* $SLURM_NTASKS_PER_NODE \* $SLURM_CPUS_PER_TASK)
		
		# Create a hostfile
		(node)$> srun hostname | sort -n > hostfile
		
		# Launch the paired end assembler:
		(node)$> abyss-pe mpirun="mpirun -x PATH -x LD_LIBRARY_PATH -hostfile hostfile" name=${ABYSSNAME} np=${ABYSSNPROC} k=31 n=10 lib=pairedend pairedend="${ABYSSINPDIR}/SRR001666_1.fastq.bz2 ${ABYSSINPDIR}/SRR001666_2.fastq.bz2" > ${ABYSSNAME}.out 2> ${ABYSSNAME}.err

**Question: Why do we use the -x VARIABLE parameters for mpirun?**

Several options seen on the `abyss-pe` command line are crucial:

* we explicitly set the mpirun command
  - we export several environment variables to all the remote nodes, otherwise required paths (for the binaries, libraries) would not be known by the MPI processes running there
* we do not specify `-np $ABYSSNPROC` in the mpirun command, as it set with `abyss-pe`'s np parameter and internally passed on to mpirun

The execution should take around 12 minutes, meanwhile we can check its progress by monitoring the .out/.err output files:

    (access)$> tail -f ~/bioinfo-tutorial/abyss/abysstest.*
    # We exit the tail program with CTRL-C

On **Gaia**, we can also connect to the job (recall oarsub -C $JOBID) from a different terminal or Screen window and see the different ABySS phases with `htop`.

Because the `abyss-pe` workflow (pipeline) includes several processing steps with different applications of which only ABYSS-P is MPI-parallel,
the speedup obtained by using more than one node will be limited to ABYSS-P's execution. Several of the other applications that are part of the
processing stages are however parallelized using OpenMP and pthreads and will thus take advantage of the cores available on the node where
`abyss-pe` was started.

The used input dataset is a well known [Illumina run of E. coli](https://trace.ddbj.nig.ac.jp/DRASearch/run?acc=SRR001666).

### Proposed exercises

Several exercises are proposed for ABySS:

1. create a launcher for ABySS using the commands shown in the previous section
2. launch jobs using 1 node: 4, 8 and 12 cores, then 2 and 4 nodes and measure the speedup obtained
3. unpack the two input files and place them on a node's /dev/shm, then rerun the experiment with 4, 8 and 12 cores and measure the speedup




## GROMACS
Characterization: CPU intensive, little I/O

### Description

__GROMACS__: GROningen MAchine for Chemical Simulations

GROMACS is a versatile package to perform molecular dynamics, i.e. simulate the Newtonian equations of motion for systems with hundreds to millions of particles.
It is primarily designed for biochemical molecules like proteins, lipids and nucleic acids that have a lot of complicated bonded interactions, but since GROMACS is extremely fast at calculating the nonbonded interactions (that usually dominate simulations) many groups are also using it for research on non-biological systems, e.g. polymers [\[\*\]](http://www.gromacs.org/About_Gromacs).

### Example

This example will be ran in an interactive session, with batch-mode executions
being proposed later on as exercises.

* Gaia

		# Connect to Gaia (Linux/OS X):
		(yourmachine)$> ssh access-gaia.uni.lu
		
		# Request 1 full node in an interactive job:
		(access-gaia)$> oarsub -I -l nodes=1,walltime=00:30:00

* Iris

		# Connect to Iris (Linux/OS X):
		(yourmachine)$> ssh access-iris.uni.lu
		
		# Request half a node in an interactive job:
		(access-iris)$> srun -p interactive --qos qos-interactive -t 0-0:30:0 -N 1 -c 1 --ntasks-per-node=14 --pty bash


```
# Check the GROMACS versions installed on the clusters:
(node)$> module avail 2>&1 | grep -i gromacs
```

There used to be two versions of GROMACS available on Gaia, a `hybrid` and a `mt` version

  - the hybrid version is OpenMP and MPI-enabled, all binaries have a '\_mpi' suffix
  - the mt version is only OpenMP-enabled, as such it can take advantage of only one node's cores (however it may be faster on
single-node executions than the hybrid version)

Currently only the following version of GROMACS is available:

* bio/GROMACS/2016.3-intel-2017a-hybrid

We will perform our tests with the hybrid version:

```
# Load the MPI-enabled Gromacs, without CUDA support:
(node)$> module load bio/GROMACS

# Check that it has been loaded, along with its dependencies:
(node)$> module list

# Check the capabilities of the mdrun binary, note its suffix:
(node)$> gmx_mpi -version 2>/dev/null

# Go to the test directory
(node)$> cd ~/bioinfo-tutorial/gromacs

# Set the number of OpenMP threads to 1
(node)$> export OMP_NUM_THREADS=1
```

* Gaia
	
		# Perform a position restrained Molecular Dynamics run
		(node)$> mpirun -np 12 -hostfile $OAR_NODEFILE -envlist OMP_NUM_THREADS,PATH,LD_LIBRARY_PATH gmx_mpi mdrun -v -s pr -e pr -o pr -c after_pr -g prlog > test.out 2>&1

* Iris

		# Perform a position restrained Molecular Dynamics run
		(node)$> srun -n 12 gmx_mpi mdrun -v -s pr -e pr -o pr -c after_pr -g prlog > test.out 2>&1

We notice here that we are running `gmx_mpi` in parallel with mpirun/srun on 12/14 cores, and we explicitly export the OMP_NUM_THREADS
variable to any remote node such that only one thread per MPI process will be created.

**Question: What will happen if we do not set the number of OpenMP threads to 1?**

GROMACS has many parallelization options and several parameters can be tuned to give you better performance depending on your workflow, see the references in the last section of this tutorial.

The used input corresponds to the [Ribonuclease S-peptide](http://manual.gromacs.org/archive/4.6.5/online/speptide.html) example,
which has been changed to perform 50k steps in the Molecular Dynamics run with position restraints on the peptide.

### Proposed exercises

Several exercises are proposed for GROMACS:

1. create a launcher for GROMACS using the commands shown in the previous section
2. launch jobs using 1 node: 1, 2, 4, 8, 10 and 12 cores and measure the speedup obtained
3. check what happens when executing mdrun with 16 and 24 cores
4. launch a job using one full node that has GPU cards and run the GPU-enabled GROMACS to see if a speedup is obtained




## Bowtie2/TopHat
Characterization: data intensive, RAM intensive

### Description

__Bowtie2__: Fast and sensitive read alignment

Bowtie 2 is an ultrafast and memory-efficient tool for aligning sequencing reads to long reference sequences. It is particularly good at aligning reads of about 50 up to 100s or 1,000s of characters, and particularly good at aligning to relatively long (e.g. mammalian) genomes [\[\*\]](http://bowtie-bio.sourceforge.net/bowtie2/index.shtml).

__TopHat__ : A spliced read mapper for RNA-Seq

TopHat is a program that aligns RNA-Seq reads to a genome in order to identify exon-exon splice junctions. It is built on the ultrafast short read mapping program Bowtie [\[\*\]](http://ccb.jhu.edu/software/tophat/index.shtml).

### Example

This example will show you how to use the latest version of TopHat in conjunction with the latest Bowtie2, by using the
versions prebuilt for Linux by the developers.

* Gaia

		# Connect to Gaia (Linux/OS X):
		(yourmachine)$> ssh access-gaia.uni.lu
		
		# Request 1 full node in an interactive job:
		(gaia-frontend)$> oarsub -I -l nodes=1,walltime=00:30:00

* Iris

		# Connect to Iris (Linux/OS X):
		(yourmachine)$> ssh access-iris.uni.lu
		
		# Request half a node in an interactive job:
		(access-iris)$> srun -p interactive --qos qos-interactive -t 0-0:30:0 -N 1 -c 14 --ntasks-per-node=1 --pty bash

```
# Create a folder for the new software and go to it
(node)$> mkdir ~/bioinfo-tutorial/newsoft
(node)$> cd ~/bioinfo-tutorial/newsoft

# Download latest Bowtie2 and Tophat, plus the SAM tools dependency:
(node)$> wget https://downloads.sourceforge.net/project/bowtie-bio/bowtie2/2.3.4.1/bowtie2-2.3.4.1-linux-x86_64.zip
(node)$> wget http://ccb.jhu.edu/software/tophat/downloads/tophat-2.1.1.Linux_x86_64.tar.gz
(node)$> wget https://github.com/samtools/samtools/releases/download/1.8/samtools-1.8.tar.bz2

# Unpack the three archives
(node)$> unzip bowtie2-2.3.4.1-linux-x86_64.zip
(node)$> tar xzvf tophat-2.1.1.Linux_x86_64.tar.gz
(node)$> tar xjvf samtools-1.8.tar.bz2

# SAMtools requires compilation:
(node)$> module load tools/bzip2/1.0.6-intel-2017a
(node)$> cd samtools-1.8 && ./configure && make && cd ..

# Create a file containing the paths to the binaries, to be sourced when needed
(node)$> echo "export PATH=$HOME/bioinfo-tutorial/newsoft/bowtie2-2.3.4.1-linux-x86_64:\$PATH" > newsoft
(node)$> echo "export PATH=$HOME/bioinfo-tutorial/newsoft/tophat-2.1.1.Linux_x86_64:\$PATH" >> newsoft
(node)$> echo "export PATH=$HOME/bioinfo-tutorial/newsoft/samtools-1.8:\$PATH" >> newsoft
(node)$> source newsoft

# You can now check that both main applications can be run:
(node)$> bowtie2 --version
(node)$> tophat2 --version
```

Now we will make a quick TopHat test, using the provided sample files:

```
# Go to the test directory, unpack the sample dataset and go to it
(node)$> cd ~/bioinfo-tutorial/tophat
(node)$> tar xzvf test_data.tar.gz
(node)$> cd test_data


# Launch TopHat, with Bowtie2 in serial mode
(node)$> tophat -r 20 test_ref reads_1.fq reads_2.fq

# Launch TopHat, with Bowtie2 in parallel mode
(node)$> tophat -p 12 -r 20 test_ref reads_1.fq reads_2.fq
```

We can see that for this fast execution, increasing the number of threads does not improve the calculation time due to the relatively high overhead of thread creation.
Note that TopHat / Bowtie are not MPI applications and as such can take advantage of at most one compute node.

Next, we will make a longer test, where it will be interesting to monitor the TopHat pipeline (with `htop` for example) to see the transitions between the serial
and parallel stages (left as an exercise).

```
# Load the file which will export $TOPHATTEST2 in the environment
(node)$> source ~/bioinfo-tutorial/tophat/test2_path

# Launch TopHat, with Bowtie2 in parallel mode
(node)$> tophat2 -p 12 -g 1 -r 200 --mate-std-dev 30 -o ./  $TOPHATTEST2/chr10.hs $TOPHATTEST2/SRR027888.SRR027890_chr10_1.fastq $TOPHATTEST2/SRR027888.SRR027890_chr10_2.fastq
```

The input data for the first test corresponds to the [TopHat test set](http://ccb.jhu.edu/software/tophat/tutorial.shtml),
while the second test is an example of aligning reads to the chromosome 10 of the human genome [as given here](http://www.bigre.ulb.ac.be/courses/statistics_bioinformatics/practicals/ASG1_2012/rnaseq_td/rnaseq_td.html).

### Proposed exercises

The following exercises are proposed for TopHat/Bowtie2:

1. create a launcher for TopHat using the commands shown in the previous section
2. launch jobs with 1, 2, 4, 8 and 10 cores on one node, using the second test files, and measure the speedup obtained




## mpiBLAST
Characterization: data intensive, little RAM overhead, native parallelization

### Description

__mpiBLAST__: Open-Source Parallel BLAST

mpiBLAST is a freely available, open-source, parallel implementation of NCBI BLAST. By efficiently utilizing distributed computational resources through database fragmentation, query segmentation, intelligent scheduling, and parallel I/O, mpiBLAST improves NCBI BLAST performance by several orders of magnitude while scaling to hundreds of processors  [\[\*\]](http://www.mpiblast.org/).

### Example

This example will be ran in an interactive session, with batch-mode executions
being proposed later on as exercises.

* Gaia

		# Connect to Gaia (Linux/OS X):
		(yourmachine)$> ssh access-gaia.uni.lu
		
		# Request 1 full node in an interactive job:
		(access-gaia)$> oarsub -I -l nodes=1,walltime=00:30:00
		
		# Load the bioinfo software set
		(node)$> module use $RESIF_ROOTINSTALL/bioinfo/modules/all

* Iris

		# Connect to Iris (Linux/OS X):
		(yourmachine)$> ssh access-iris.uni.lu
		
		# Request half a node in an interactive job:
		(access-iris)$> srun -p interactive --qos qos-interactive -t 0-0:30:0 -N 1 -c 1 --ntasks-per-node=14 --pty bash
		
		# Load the bioinfo software set
		(node)$> module use /opt/apps/resif/data/stable/bioinfo/modules/all

```
# Check the mpiBLAST versions installed on the clusters:
(node)$> module avail 2>&1 | grep -i mpiblast

# Load the default mpiBLAST version:
(node)$> module load bio/mpiBLAST

# Check that it has been loaded, along with its dependencies:
(node)$> module list

# The mpiBLAST binaries should now be in your path
(node)$> mpiformatdb --version
(node)$> mpiblast --version
```


mpiBLAST requires access to NCBI substitution matrices and pre-formatted BLAST databases. For the purposes of this tutorial, a FASTA (NR)
database has been formatted and split into 12 fragments, enabling the parallel alignment of a query against the database.

A `.ncbirc` file containing the paths to the necessary data files can be downloaded from [here](https://raw.github.com/ULHPC/tutorials/devel/bio/basics/mpiblast/.ncbirc)
and placed in your `$HOME` directory (make sure to backup an existing `$HOME/.ncbirc` before overwriting it with the one in this tutorial).

**Question: Knowing that the databases can take tens of gigabytes, what is an appropriate storage location for them on the clusters?**

We will run a test using mpiBLAST. Note that mpiBLAST requires running with at least 3 processes, 2 dedicated for scheduling tasks and
coordinating file output, with the additional processes performing the search.

* Gaia

		# Go to the test directory and execute mpiBLAST with one core for search
		(node)$> cd ~/bioinfo-tutorial/mpiblast
		(node)$> mpirun -np 3 mpiblast -p blastp -d nr -i test.fa -o test.out
		
		# Note the speedup when using 12 cores
		(node)$> mpirun -np 12 mpiblast -p blastp -d nr -i test.fa -o test.out

* Iris

		# Go to the test directory and execute mpiBLAST with one core for search
		(node)$> cd ~/bioinfo-tutorial/mpiblast
		(node)$> srun -n 3 mpiblast -p blastp -d nr -i test.fa -o test.out
		
		# Note the speedup when using 14 cores
		(node)$> srun -n 14 mpiblast -p blastp -d nr -i test.fa -o test.out

### Proposed exercises

The following exercises are proposed for mpiBLAST:

1. create a launcher for mpiBLAST, making sure to export the required environment to the remote nodes
2. launch jobs with 8, 14 and 24 cores across two nodes and measure the speedup obtained


## Useful references

  - [Gromacs parallelization](http://www.gromacs.org/Documentation/Acceleration_and_parallelization)
  - [Gromacs GPU acceleration](http://www.gromacs.org/GPU_acceleration)
  - [Gromacs USA workshop](http://www.gromacs.org/Documentation/Tutorials/GROMACS_USA_Workshop_and_Conference_2013)
  - [Tutorial on GROMACS parallelization schemes](http://www.gromacs.org/Documentation/Tutorials/GROMACS_USA_Workshop_and_Conference_2013/Parallelization_schemes_and_GPU_acceleration%3a_Szilard_Pall%2c_Session_2B)
