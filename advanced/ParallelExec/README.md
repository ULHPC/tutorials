`README.md`

Copyright (c) 2015 Valentin Plugaru <Valentin.Plugaru@uni.lu>

-------------------


# UL HPC Tutorial: Running parallel software: test cases on CFD / MD / Chemistry applications

The objective of this session is to exemplify the execution of several common, parallel, Computational Fluid Dynamics, Molecular Dynamics and Chemistry software on the [UL HPC](http://hpc.uni.lu) platform.

Targeted applications include:

* [OpenFOAM](http://www.openfoam.org): CFD package for solving complex fluid flows involving chemical reactions, turbulence and heat transfer
* [NAMD](http://www.ks.uiuc.edu/Research/namd): parallel molecular dynamics code designed for high-performance simulation of large biomolecular systems
* [ASE](https://wiki.fysik.dtu.dk/ase): Atomistic Simulation Environment (Python-based) with the aim of setting up, steering, and analyzing atomistic simulations
* [ABINIT](http://www.abinit.org): materials science package implementing DFT, DFPT, MBPT and TDDFT
* [Quantum Espresso](http://www.quantum-espresso.org): integrated suite of tools for electronic-structure calculations and materials modeling at the nanoscale


The tutorial will cover:

1. OAR basics for parallel execution
2. different MPI suites available on UL HPC
3. running simple test cases in parallel
4. running QuantumEspresso in parallel over a single node and over multiple nodes
5. running OpenFOAM in parallel over a single node and over multiple nodes
6. running ABINIT in parallel over a single node and over multiple nodes
7. the interesting case of the ASE toolkit

## Prerequisites

As part of this tutorial several input files will be required and you will need to download them 
before following the instructions in the next sections:

Or simply clone the full tutorials repository and make a link to this tutorial

        (gaia-frontend)$> git clone https://github.com/ULHPC/tutorials.git
        (gaia-frontend)$> ln -s tutorials/advanced/ParallelExec/ ~/parallelexec-tutorial

## Basics

### OAR basics for parallel execution

First of all, we will submit a job with 2 cores on each of 2 compute nodes for 1 hour.  
Please note that the Gaia cluster can be replaced at any point in this tutorial with the Chaos cluster if not enough resources are immediately available.

       (gaia-frontend)$> oarsub -I -l nodes=2/core=2,walltime=1 
       (node)$> 

The OAR scheduler provides several environment variables once we are inside a job, check them out with

       (node)$> env | grep OAR_

We are interested especially in the environment variable which points to the file containing the list of hostnames reserved for the job.  
This variable is OAR\_NODEFILE, yet there are several others pointing to the same (OAR\_NODE\_FILE, OAR\_FILE\_NODES and OAR_RESOURCE_FILE).  
Let's check its content:

       (node)$> cat $OAR_NODEFILE

To get the number of cores available in the job, we can use the wordcount `wc` utility, in line counting mode:

       (node)$> cat $OAR_NODEFILE | wc -l

### MPI suites available on the platform

Now, let's check for the environment modules (available through Lmod) which match MPI (Message Passing Interface) the libraries that provide inter-process communication over a network:

       (node)$> module avail mpi/
       
       ---------------------------------------------------- /opt/apps/resif/devel/v1.1-20150414/core/modules/mpi ----------------------------------------------------
          mpi/OpenMPI/1.6.4-GCC-4.7.2    mpi/OpenMPI/1.8.4-GCC-4.9.2 (D)    mpi/impi/4.1.0.030-iccifort-2013.3.163    mpi/impi/5.0.3.048-iccifort-2015.3.187 (D)
       
       ------------------------------------------------- /opt/apps/resif/devel/v1.1-20150414/core/modules/toolchain -------------------------------------------------
          toolchain/gompi/1.4.10    toolchain/iimpi/5.3.0    toolchain/iimpi/7.3.5 (D)
       
         Where:
          (D):  Default Module
       
       Use "module spider" to find all possible modules.
       Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".


Perform the same search for the toolchains:

       (node)$> module avail toolchain/
 
Toolchains represent sets of compilers together with libraries commonly required to build software, such as MPI, BLAS/LAPACK (linear algebra) and FFT (Fast Fourier Transforms).
For more details, see [the EasyBuild page on toolchains](https://github.com/hpcugent/easybuild/wiki/Compiler-toolchains).

For our initial tests we will use the __goolf__ toolchain which includes GCC, OpenMPI, OpenBLAS/LAPACK, ScaLAPACK(/BLACS) and FFTW:

       (node)$> module load toolchain/goolf/1.4.10
       (node)$> module list

The main alternative to this toolchain (as of June 2015) is __ictce__ (toolchain/ictce/7.3.5) that includes the Intel tools icc, ifort, impi and imkl.

### Simple test cases

We will now try to run the `hostname` application, which simply shows a system's host name.   
Check out the differences (if any) between the following executions:

       (node)$> hostname
       (node)$> mpirun hostname
       (node)$> mpirun -n 2 hostname
       (node)$> mpirun -np 2 hostname
       (node)$> mpirun -n 4 hostname
       (node)$> mpirun -hostfile $OAR_NODEFILE hostname
       (node)$> mpirun -hostfile $OAR_NODEFILE -n 2 hostname
       (node)$> mpirun -hostfile $OAR_NODEFILE -n 3 hostname
       (node)$> mpirun -hostfile $OAR_NODEFILE -npernode 1 hostname

Note that the `hostname` application is _not_ a parallel application, with MPI we are simply launching it on the different nodes available in the job.  

Now, we will compile and run a simple MPI application. Save the following source code in /tmp/hellompi.c :

       ### hellompi.c
       #include <mpi.h>
       #include <stdio.h>
       #include <stdlib.h>
       int main(int argc, char** argv) {
         MPI_Init(NULL, NULL);
         int world_size;
         MPI_Comm_size(MPI_COMM_WORLD, &world_size);
         int world_rank;
         MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
         char processor_name[MPI_MAX_PROCESSOR_NAME];
         int name_len;
         MPI_Get_processor_name(processor_name, &name_len);
         printf("Hello world from %s, rank %d out of %d CPUs\n", processor_name, world_rank, world_size);
         MPI_Finalize();
       }

Compile it:

       (node)$> cd /tmp
       (node)$> mpicc -o hellompi hellompi.c

Run it:

       (node)$> mpirun -hostfile $OAR_NODEFILE ./hellompi

Why didn't it work? Remember we stored this application on a compute node's /tmp directory, which is local to each node, not shared.  
Thus the application couldn't be found (more on this later) on the remote nodes.

Let's move it to the `$HOME` directory which is common across the cluster, and try again:

       (node)$> mv /tmp/hellompi ~/parallelexec-tutorial
       (node)$> cd ~/parallelexec-tutorial
       (node)$> mpirun -hostfile $OAR_NODEFILE ./hellompi

Now some different error messages are shown, about loading shared libraries, and the execution hangs (stop it with Ctrl-C). Why?
Remember that on the current node we have loaded the goolf toolchain module, which has populated the environment with important details such as the paths  to applications and libraries. This environment is not magically synchronized across the multiple nodes in our OAR job, thus when the hellompi process is started remotely, some libraries are not found.

We will explicitly tell mpirun to export two important environment variables to the remote nodes:

       (node)$> mpirun -x PATH -x LD_LIBRARY_PATH -hostfile $OAR_NODEFILE ./hellompi

Now it (should have) worked and we can try different executions: 

       (node)$> mpirun -x PATH -x LD_LIBRARY_PATH ./hellompi
       (node)$> mpirun -x PATH -x LD_LIBRARY_PATH -n 2 ./hellompi
       (node)$> mpirun -x PATH -x LD_LIBRARY_PATH -np 2 ./hellompi
       (node)$> mpirun -x PATH -x LD_LIBRARY_PATH -n 4 ./hellompi
       (node)$> mpirun -x PATH -x LD_LIBRARY_PATH -hostfile $OAR_NODEFILE ./hellompi
       (node)$> mpirun -x PATH -x LD_LIBRARY_PATH -hostfile $OAR_NODEFILE -n 2 ./hellompi
       (node)$> mpirun -x PATH -x LD_LIBRARY_PATH -hostfile $OAR_NODEFILE -n 3 ./hellompi
       (node)$> mpirun -x PATH -x LD_LIBRARY_PATH -hostfile $OAR_NODEFILE -npernode 1 ./hellompi

At the end of these tests, we clean the environment by running `module purge`:

       (node)$> module purge
       (node)$> module list

## QuantumESPRESSO

Check for the available versions of QuantumESPRESSO (QE in short), as of June 2015 this shows:

       (node)$> module spider quantum
       
       ----------------------------------------------------------------------------------------------------------------------------------------------------------
         chem/QuantumESPRESSO:
       ----------------------------------------------------------------------------------------------------------------------------------------------------------
           Description:
             Quantum ESPRESSO is an integrated suite of computer codes for electronic-structure calculations and materials modeling at the nanoscale. It is
             based on density-functional theory, plane waves, and pseudopotentials (both norm-conserving and ultrasoft). - Homepage: http://www.pwscf.org/ 
       
            Versions:
               chem/QuantumESPRESSO/5.0.2-goolf-1.4.10-hybrid
               chem/QuantumESPRESSO/5.0.2-goolf-1.4.10
               chem/QuantumESPRESSO/5.0.2-ictce-5.3.0-hybrid
               chem/QuantumESPRESSO/5.0.2-ictce-5.3.0
               chem/QuantumESPRESSO/5.1.2-ictce-7.3.5

One thing we note is that some versions have a _-hybrid_ suffix. These versions are hybrid MPI+OpenMP builds of QE.  
MPI+OpenMP QE can give better performance than the pure MPI versions, by running only one MPI process per node (instead of one MPI process for each core in the job) that creates (OpenMP) threads which run in parallel locally.

Load the latest QE version:

       (node)$> module load chem/QuantumESPRESSO/5.1.2-ictce-7.3.5

We will use the PWscf (Plane-Wave Self-Consistent Field) package of QE for our tests.
Run it in sequential mode, it will wait for your input. You should see a "Parallel version (MPI), running on     1 processors" message, and can stop it with CTRL-C:

       (node)$> pw.x
       (node)$> mpirun -n 1 pw.x 

Now try the parallel run over all the nodes/cores in the job:

       (node)$> mpirun -hostfile $OAR_NODEFILE pw.x

Before stopping it, check that pw.x processes are created on the remote node. You will need to:

1. open a second connection to the cluster, or a second window if you're using `screen` or `tmux`
2. connect to the job with `oarsub -C $JOBID`
3. connect from the head node of the job to the remote job with `oarsh $hostname`
4. use `htop` to show the processes, filter the shown list to see only your user with `u` and then selecting your username

Note that this check procedure can be invaluable when you are running an application for the first time, or with new options.  
Generally, some things to look for are:

* that processes _are_ created on the remote node, instead of all of them on the head node (which leads to huge slowdowns)
* the percentage of CPU usage those processes have, for CPU-intensive work, the values in the CPU% column should be close to 100%
  - if the values are constantly close to 50%, or 25% (or even less) it may mean that more parallel processes were started than should have on that node (e.g. if all processes are running on the head node) and that they are constantly competing for the same cores, which makes execution very slow
* the number of threads created by each process
  - here the number of OpenMP threads, controlled through the OMP\_NUM\_THREADS environment variable or Intel MKL threads (MKL\_NUM\_THREADS) may need to be tuned

Now we will run `pw.x` to perform electronic structure calculations in the presence of a finite homogeneous electric field, and we will use sample input (PW example10) to calculate high-frequency dielectric constant of bulk Silicon.
For reference, many examples are given in the installation directory of QE, see `$EBROOTQUANTUMESPRESSO/espresso-$EBVERSIONQUANTUMESPRESSO/PW/examples`.

       (node)$> cd ~/parallelexec-tutorial 
       (node)$> cd inputs/qe 
       (node)$> pw.x < si.scf.efield2.in

We will see the calculation progress, this serial execution should take around 2 minutes.

Next, we will clean up the directory holding output files, and re-run the example in parallel:

       (node)$> rm -rf out
       (node)$> mpirun -hostfile $OAR_NODEFILE pw.x < si.scf.efield2.in > si.scf.efield2.out

When the execution ends, we can take a look at the last 10 lines of output and check the execution time:

       (node)$> tail si.scf.efield2.out

You can now try to run the same examples but with the `chem/QuantumESPRESSO/5.0.2-goolf-1.4.10-hybrid` module.
Things to test:

- basic execution vs usage of the `npernode` parameter of OpenMPI's mpirun
- explicitly setting the number of OpenMP threads
- increasing the number of OpenMP threads

### References

  - [QE: user's guide](www.quantum-espresso.org/wp-content/uploads/Doc/user_guide.pdf)
  - [QE: understanding parallelism](http://www.quantum-espresso.org/wp-content/uploads/Doc/user_guide/node16.html)
  - [QE: running on parallel machines](http://www.quantum-espresso.org/wp-content/uploads/Doc/user_guide/node17.html) 
  - [QE: parallelization levels](http://www.quantum-espresso.org/wp-content/uploads/Doc/user_guide/node18.html)
