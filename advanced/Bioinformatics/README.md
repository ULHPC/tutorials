`README.md`

Copyright (c) 2014 Valentin Plugaru <Valentin.Plugaru@gmail.com>

-------------------


# UL HPC Tutorial: Bioinformatics software on the UL HPC platform

The objective of this tutorial is to exemplify the execution of several
Bioinformatics packages on top of the [UL HPC](http://hpc.uni.lu) platform.

The targeted applications are:

* [ABySS](http://www.bcgsc.ca/platform/bioinfo/software/abyss)
* [Gromacs](http://www.gromacs.org/)
* [Bowtie2](http://bowtie-bio.sourceforge.net/bowtie2/index.shtml) / [TopHat](http://tophat.cbcb.umd.edu/)

The tutorial will:

1. show you how to load and run pre-configured versions of these applications on the clusters
2. show you how to download and use updated versions of Bowtie2/TopHat
3. discuss the parallelization capabilities of these applications

## Prerequisites

This tutorial relies on several input files for the bioinformatics packages, thus you will need to download them
before following the instructions in the next sections:

        (gaia-frontend)$> mkdir -p ~/bioinfo-tutorial/gromacs ~/bioinfo-tutorial/tophat 
        (gaia-frontend)$> cd ~/bioinfo-tutorial
        (gaia-frontend)$> wget https://raw.github.com/ULHPC/tutorials/devel/advanced/Bioinformatics/gromacs/pr.tpr -O gromacs/pr.tpr
        (gaia-frontend)$> wget https://raw.github.com/ULHPC/tutorials/devel/advanced/Bioinformatics/tophat/test_data.tar.gz -O tophat/test_data.tar.gz

Or simply clone the full tutorials repository and make a link to the Bioinformatics tutorial:

        (gaia-frontend)$> git clone https://github.com/ULHPC/tutorials.git
        (gaia-frontend)$> ln -s tutorials/advanced/Bioinformatics/ ~/bioinfo-tutorial

        
## ABySS

### Description

__ABySS__: Assembly By Short Sequences

ABySS is a de novo, parallel, paired-end sequence assembler that is designed for short reads. 
The single-processor version is useful for assembling genomes up to 100 Mbases in size. 
The parallel version is implemented using MPI and is capable of assembling larger genomes [\[\*\]](http://www.bcgsc.ca/platform/bioinfo/software/abyss).

### Example

This example will be ran in an [interactive OAR session](https://hpc.uni.lu/users/docs/oar.html#concepts), with batch-mode executions
being proposed later on as exercises.

        # Connect to Gaia (Linux/OS X):
        (yourmachine)$> ssh access-gaia.uni.lu
     
        # Request 1 full node in an interactive job:
        (gaia-frontend)$> oarsub -I -l node=1,walltime=00:30:00

        # Check the ABySS versions installed on the clusters:
        (node)$> module available 2>&1 | grep -i abyss
        
        # Load a specific ABySS version:
        (node)$> module load ABySS/1.3.4-goolf-1.4.10-Python-2.7.3

        # Check that it has been loaded, along with its dependencies:
        (node)$> module list
        
        # All the ABySS binaries are now in your path (check with TAB autocompletion)
        (node)$> abyss-<TAB><TAB>

In the ABySS package only the `ABYSS-P` application is parallelized using MPI and can be run on several cores (and across several nodes) using
the `abyss-pe` launcher.

        # Create a test directory and go to it
        (node)$> mkdir ~/bioinfo-tutorial/abyss
        (node)$> cd ~/bioinfo-tutorial/abyss
       
        # Set the input files' directory in the environment
        (node)$> export ABYSSINPDIR=/scratch/users/vplugaru/bioinfo-inputs/abyss
        
        # Give a name to the experiment
        (node)$> export ABYSSNAME='abysstest'
        
        # Set the number of cores to use based on OAR's host file
        (node)$> export ABYSSNPROC=$(cat $OAR_NODEFILE | wc -l)
        
        # Launch the paired end assembler:
        (node)$> abyss-pe mpirun="mpirun -x PATH -x LD_LIBRARY_PATH -hostfile $OAR_NODEFILE" name=${ABYSSNAME} np=${ABYSSNPROC} k=31 n=10 lib=pairedend pairedend="${ABYSSINPDIR}/SRR001666_1.fastq.bz2 ${ABYSSINPDIR}/SRR001666_2.fastq.bz2" > ${ABYSSNAME}.out 2> ${ABYSSNAME}.err
 
Several options seen on the `abyss-pe` command line are crucial:

* we explicitly set the mpirun command
  - we export several environment variables to all the remote nodes, otherwise required paths (for the binaries, libraries) would not be known by the MPI processes running there
* we do not specify `-np $ABYSSNPROC` in the mpirun command, as it set with `abyss-pe`'s np parameter and internally passed on to mpirun

The execution should take around 12 minutes, meanwhile we can check its progress by monitoring the .out/.err output files:

         (gaia-frontend)$> tail -f ~/bioinfo-tutorial/abyss/abysstest.*
         # We exit the tail program with CTRL-C

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

### Description

__GROMACS__: GROningen MAchine for Chemical Simulations

GROMACS is a versatile package to perform molecular dynamics, i.e. simulate the Newtonian equations of motion for systems with hundreds to millions of particles.
It is primarily designed for biochemical molecules like proteins, lipids and nucleic acids that have a lot of complicated bonded interactions, but since GROMACS is extremely fast at calculating the nonbonded interactions (that usually dominate simulations) many groups are also using it for research on non-biological systems, e.g. polymers [\[\*\]](http://www.gromacs.org/About_Gromacs).

### Example

This example will be ran in an [interactive OAR session](https://hpc.uni.lu/users/docs/oar.html#concepts), with batch-mode executions
being proposed later on as exercises.

        # Connect to Gaia (Linux/OS X):
        (yourmachine)$> ssh access-gaia.uni.lu
     
        # Request 1 full node in an interactive job:
        (gaia-frontend)$> oarsub -I -l node=1,walltime=00:30:00

        # Check the GROMACS versions installed on the clusters:
        (node)$> module available 2>&1 | grep -i gromacs
        
Several GROMACS builds are available, we will focus only on the ones corresponding to the version 4.6.5:

* GROMACS/4.6.5-goolf-1.4.10-hybrid,  GROMACS/4.6.5-goolfc-2.6.10-hybrid
* GROMACS/4.6.5-goolf-1.4.10-mt, GROMACS/4.6.5-goolfc-2.6.10-mt

We notice here two details:

* the `goolf` and `goolfc` toolchains used to compile GROMACS
  - the major difference here is that `goolfc` contains CUDA support
* the `hybrid` and `mt` versions
  - the hybrid version is OpenMP and MPI-enabled, all binaries have a '\_mpi' suffix
  - the mt version is only OpenMP-enabled, as such it can take advantage of only one node's cores (however it may be faster on
single-node executions than the hybrid version)

We will perform our tests with the hybrid version:

        # Load the MPI-enabled Gromacs, without CUDA support:
        (node)$> module load GROMACS/4.6.5-goolf-1.4.10-hybrid

        # Check that it has been loaded, along with its dependencies:
        (node)$> module list
        
        # Check the capabilities of the mdrun binary, note its suffix:
        (node)$> mdrun_mpi -version 2>/dev/null

        # Create a test directory and go to it
        (node)$> mkdir ~/bioinfo-tutorial/gromacs
        (node)$> cd ~/bioinfo-tutorial/gromacs
       
        # Copy the input file to the current directory
        (node)$> cp /scratch/users/vplugaru/bioinfo-inputs/gromacs/pr.tpr .
        
        # Set the number of OpenMP threads to 1
        (node)$> export OMP_NUM_THREADS=1
        
        # Perform a position restrained Molecular Dynamics run
        (node)$> mpirun -np 12 -hostfile $OAR_NODEFILE -x OMP_NUM_THREADS -x PATH -x LD_LIBRARY_PATH mdrun_mpi -v -s pr -e pr -o pr -c after_pr -g prlog > test.out 2>&1
        
We notice here that we are running `mdrun_mpi` in parallel with mpirun on 12 cores, and we explicitly export the OMP_NUM_THREADS
variable to any remote nodes such that only one thread per MPI process will be created.
GROMACS , see also the references in the last section of this tutorial.
 
The used input corresponds to the [Ribonuclease S-peptide](http://manual.gromacs.org/online/speptide.html) example,
which has been changed to perform 50k steps in the Molecular Dynamics run with position restraints on the peptide.

### Proposed exercises

Several exercises are proposed for GROMACS:

1. create a launcher for GROMACS using the commands shown in the previous section
2. launch jobs using 1 node: 1, 2, 4, 8, 10 and 12 cores and measure the speedup obtained
3. check what happens when running mdrun with 16 and 24 cores



## Bowtie2/TopHat

### Description

__Bowtie2__: Fast and sensitive read alignment

Bowtie 2 is an ultrafast and memory-efficient tool for aligning sequencing reads to long reference sequences. It is particularly good at aligning reads of about 50 up to 100s or 1,000s of characters, and particularly good at aligning to relatively long (e.g. mammalian) genomes [\[\*\]](http://bowtie-bio.sourceforge.net/bowtie2/index.shtml).

__TopHat__ : A spliced read mapper for RNA-Seq

TopHat is a program that aligns RNA-Seq reads to a genome in order to identify exon-exon splice junctions. It is built on the ultrafast short read mapping program Bowtie [\[\*\]](http://tophat.cbcb.umd.edu/manual.shtml#whis).

### Example

This example will show you how to use the latest version of TopHat in conjunction with the latest Bowtie2, by using the 
versions prebuilt for Linux by the developers.


        # Connect to Gaia (Linux/OS X):
        (yourmachine)$> ssh access-gaia.uni.lu
     
        # Request 1 full node in an interactive job:
        (gaia-frontend)$> oarsub -I -l node=1,walltime=00:30:00

        # Create a folder for the new software and go to it
        (node)$> mkdir $WORK/newsoft
        (node)$> cd $WORK/newsoft
        
        # Download latest Bowtie2 and Tophat, plus the SAM tools dependency:
        (node)$> wget http://downloads.sourceforge.net/project/bowtie-bio/bowtie2/2.2.2/bowtie2-2.2.2-linux-x86_64.zip
        (node)$> wget http://tophat.cbcb.umd.edu/downloads/tophat-2.0.11.Linux_x86_64.tar.gz
        (node)$> wget http://downloads.sourceforge.net/project/samtools/samtools/0.1.19/samtools-0.1.19.tar.bz2
        
        # Unpack the three archives
        (node)$> unzip bowtie2-2.2.2-linux-x86_64.zip
        (node)$> tar xzvf tophat-2.0.11.Linux_x86_64.tar.gz
        (node)$> tar xjvf samtools-0.1.19.tar.bz2
        
        # SAM tools requires compilation
        (node)$> cd samtools-0.1.19 && make && cd ..
        
        # Create a file containing the paths to the binaries, to be sourced when needed
        (node)$> echo "export PATH=$WORK/newsoft/bowtie2-2.2.2:\$PATH" > newsoft
        (node)$> echo "export PATH=$WORK/newsoft/tophat-2.0.11.Linux_x86_64:\$PATH" >> newsoft
        (node)$> echo "export PATH=$WORK/newsoft/samtools-0.1.19:\$PATH" >> newsoft
        (node)$> source newsoft
        
        # You can now check that both main applications can be run:
        (node)$> bowtie2 --version
        (node)$> tophat2 --version

Next, we will make a fast TopHat run, using the provided sample files:

        # Go to the test directory and unpack the sample dataset
        (node)$> cd ~/bioinfo-tutorial/tophat
        (node)$> tar xzvf test_data.tar.gz
        
        # Launch TopHat, with Bowtie2 in parallel mode
        (node)$> tophat -p 6 -r 20 test_ref reads_1.fq reads_2.fq

### Proposed exercises

The following exercises are proposed for TopHat/Bowtie2:

1. create a launcher for TopHat using the commands shown in the previous section
2. launch jobs with 1, 2, 4, 8, 10 and 12 cores on one node and measure the speedup obtained
        
## Useful references

  - [ABySS at SEQanswers wiki](http://seqanswers.com/wiki/ABySS)
  - [Gromacs parallelization](http://www.gromacs.org/Documentation/Acceleration_and_parallelization)
  - [Gromacs GPU acceleration](http://www.gromacs.org/GPU_acceleration)
  - [Gromacs USA workshop](http://www.gromacs.org/Documentation/Tutorials/GROMACS_USA_Workshop_and_Conference_2013)