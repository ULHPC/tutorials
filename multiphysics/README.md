[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/advanced/MultiPhysics) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/advanced/MultiPhysics/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Scalable Science - Computational Physics, Chemistry & Engineering applications

     Copyright (c) 2015-2019 UL HPC Team  <hpc-sysadmins@uni.lu>

The objective of this session is to exemplify the execution of several common, parallel, Computational Physics, Chemistry & Engineering software on the [UL HPC](http://hpc.uni.lu) platform.

Targeted applications include:

* [OpenFOAM](http://www.openfoam.org): CFD package for solving complex fluid flows involving chemical reactions, turbulence and heat transfer
* [NAMD](http://www.ks.uiuc.edu/Research/namd): parallel molecular dynamics code designed for high-performance simulation of large biomolecular systems
* [ASE](https://wiki.fysik.dtu.dk/ase): Atomistic Simulation Environment (Python-based) with the aim of setting up, steering, and analyzing atomistic simulations
* [ABINIT](http://www.abinit.org): materials science package implementing DFT, DFPT, MBPT and TDDFT
* [Quantum Espresso](http://www.quantum-espresso.org): integrated suite of tools for electronic-structure calculations and materials modeling at the nanoscale


The tutorial will cover:

1. Basics for parallel execution under the SLURM scheduler
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

        (access-iris)$> git clone https://github.com/ULHPC/tutorials.git
        (access-iris)$> ln -s tutorials/advanced/MultiPhysics/ ~/multiphysics-tutorial

## Basics

Note: you can check out either the instructions for the OAR scheduler (gaia and chaos clusters) or SLURM (for iris).

-----------------
### Preliminaries with SLURM for parallel execution

First of all, we will submit on the iris cluster an interactive job with 2 tasks on each of 2 compute nodes for 1 hour.

       (iris-frontend)$> srun -p interactive -N 2 --ntasks-per-node 2 --pty bash -i
       (node)$>

The SLURM scheduler provides several environment variables once we are inside a job, check them out with

       (node)$> env | grep SLURM_

We are interested especially in the environment variable which lists the compute nodes reserved for the job -- SLURM\_NODELIST.
Let's check its content:

       (node)$> echo $SLURM_NODELIST

To get the total number of cores available in the job, we can use the wordcount `wc` utility, in line counting mode:

       (node)$> srun hostname | wc -l

To get the number of cores available on the current compute node:

       (node)$> echo $SLURM_CPUS_ON_NODE

Some questions to think about:
- How many cores are available for the job?
- What's the difference between `-N` (nodes), `-n` (tasks) and `-c` (cores per task) when launching a job?
- What allocations can we get if we specify `-n` but not `-N`?

-----------------
### MPI suites available on the Iris cluster

Now, let's check for the environment modules (available through Lmod) which match MPI (Message Passing Interface) the libraries that provide inter-process communication over a network:

       (node)$> module avail mpi/


       ------------------------ /opt/apps/resif/data/stable/default/modules/all ------------------------
          mpi/OpenMPI/2.1.3-GCC-6.4.0-2.28    mpi/impi/2018.1.163-iccifort-2018.1.163-GCC-6.4.0-2.28    toolchain/gompi/2018a    toolchain/iimpi/2018a

       Use "module spider" to find all possible modules.
       Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".

Perform the same search for the toolchains:

       (node)$> module avail toolchain/

Toolchains represent sets of compilers together with libraries commonly required to build software, such as MPI, BLAS/LAPACK (linear algebra) and FFT (Fast Fourier Transforms).
For more details, see [the EasyBuild page on toolchains](https://github.com/hpcugent/easybuild/wiki/Compiler-toolchains).

For our initial tests we will use the __foss__ toolchain which includes GCC, __OpenMPI__, OpenBLAS/LAPACK, ScaLAPACK(/BLACS) and FFTW:

       (node)$> module load toolchain/foss
       (node)$> module list

The main alternative to this toolchain is __intel__ (toolchain/intel) that includes the Intel tools icc (C/C++ Compiler), ifort (Fortran compiler), impi (__IntelMPI__) and imkl (Math Kernel Library).

Both toolchains provide an MPI implementation, that are set up to integrate with the SLURM scheduler.
One particularity of the tight integration between the MPI libraries and SLURM is that applications can be directly started in parallel with the `srun` command, instead of the (traditional) `mpirun` or `mpiexec`.
Note that `srun` takes different parameters than either OpenMPI or IntelMPI's `mpirun`.

As of June 2019 we are testing a new set of global software with updated versions for our major applications, libraries and their dependencies.
To try them out, you'll need first to switch the software set from the production one loaded by default to the development (experimental/testing) one:

Test now:

      (node)$> module load swenv/default-env/devel
      (node)$> module avail toolchain/

To go back to the production software set:
      (node)$> module load swenv/default-env/latest

### Simple test case on iris

Now, we will compile and run a simple `hellompi` MPI application. Save the following source code in /tmp/hellompi.c :

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

Load a toolchain, which will bring in a C compiler and MPI implementation and compile the above code:

       (node)$> cd /tmp
       (node)$> mpicc -o hellompi hellompi.c

Run it:

       (node)$> srun ./hellompi

Why didn't it work? Remember we stored this application on a compute node's /tmp directory, which is local to each node, not shared.
Thus the application couldn't be found (more on this later) on the _remote_ nodes.

Let's move it to the `$HOME` directory which is common across the cluster, and try again:

       (node)$> mv /tmp/hellompi ~/multiphysics-tutorial
       (node)$> cd ~/multiphysics-tutorial
       (node)$> srun ./hellompi

Now we will run it in different ways and see what happens:

       (node)$> srun -n 1 hellompi
       (node)$> srun -n 2 hellompi
       (node)$> srun -n 3 hellompi
       (node)$> srun -n 4 hellompi
       (node)$> srun hellompi

Note that SLURM's `srun` knows the environment of your job and this will drive parallel execution, if you do not override it explicitly!

### Simple batch launchers for parallel code

Below follow example launchers that you may use for your MPI, MPI+OpenMP, or CUDA code:

* MPI only application

```bash
#!/bin/bash -l
#SBATCH -J ParallelJob
#SBATCH -n 128
#SBATCH -c 1
#SBATCH --time=0-01:00:00
#SBATCH -p batch

module load toolchain/intel
srun /path/to/your/intel-toolchain-compiled-application
```

* OpenMPI only application

```bash
#!/bin/bash -l
#SBATCH -J ThreadedJob
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 28
#SBATCH --time=0-01:00:00
#SBATCH -p batch

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
srun /path/to/your/threaded.app
```

* Multi-node hybrid application MPI+OpenMP

```bash
#!/bin/bash -l
#SBATCH -J HybridParallelJob
#SBATCH -N 10
#SBATCH --ntasks-per-node=1
#SBATCH -c 28
#SBATCH --time=0-01:00:00
#SBATCH -p batch

module load toolchain/intel
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
srun -n $SLURM_NTASKS /path/to/your/parallel-hybrid-app
```

* Multi-node multi-GPU MPI application

```bash
#!/bin/bash -l
#SBATCH -J GPUJob
#SBATCH -N 4
#SBATCH --ntasks-per-socket=4
#SBATCH -c 7
#SBATCH --gres=gpu:4
#SBATCH --time=0-01:00:00
#SBATCH -p gpu

module load toolchain/intel
module load system/CUDA

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
srun /path/to/your/gpu-app
```

## QuantumESPRESSO

Check for the available versions of QuantumESPRESSO (QE in short), as of June 2019 this shows on the Iris cluster:

       (node)$> module avail quantumespresso
       -------------------- /opt/apps/resif/data/stable/default/modules/all --------------------
          chem/QuantumESPRESSO/6.1-intel-2018a-maxter500                        phys/Yambo/4.2.2-intel-2018a-QuantumESPRESSO-6.2.1-qexml
          chem/QuantumESPRESSO/6.1-intel-2018a                                  phys/Yambo/4.2.2-intel-2018a-QuantumESPRESSO-6.2.1-qexsd_hdf5
          chem/QuantumESPRESSO/6.2.1-intel-2018a                         (D)    phys/Yambo/4.2.2-intel-2018a-QuantumESPRESSO-6.2.1-qexsd
          phys/Yambo/r15234-intel-2018a-QuantumESPRESSO-6.2.1-qexml             phys/Yambo/4.2.2-intel-2018a-QuantumESPRESSO-6.2.1
          phys/Yambo/r15234-intel-2018a-QuantumESPRESSO-6.2.1-qexsd_hdf5        phys/Yambo/4.2.4-intel-2018a-QuantumESPRESSO-6.1
          phys/Yambo/r15234-intel-2018a-QuantumESPRESSO-6.2.1-qexsd             phys/Yambo/4.3.1-r132-intel-2018aQuantumESPRESSO-6.1          (D)

See that various versions are available, and also other applications (Yambo) that are linked to specific versions of QE are found.

One thing we note is that all versions of QE are built with support for both MPI and OpenMP. In this combination QE can give better performance than the pure MPI versions, by running only one MPI process per node (instead of one MPI process for each core in the job) that creates (OpenMP) threads which run in parallel locally and communicate over shared memory.

Load the latest QE version available in the production software environment:

       (node)$> module load chem/QuantumESPRESSO

We will use the PWscf (Plane-Wave Self-Consistent Field) package of QE for our tests.
Run it in sequential mode, it will wait for your input. You should see a "Parallel version (MPI), running on     1 processors" message, and can stop it with CTRL-C:

       (node)$> pw.x
       (node)$> srun -n 1 pw.x

Now try the parallel run over all the nodes/cores in the job:

       (node)$> srun pw.x

Before stopping it, check that pw.x processes are created on the remote node. You will need to:

1. open a second connection to the cluster, or a second window if you're using `screen` or `tmux`
2. check which nodes the job is using, with `squeue -j $JOBID`
3. connect to the job on the second node from the set of nodes from the step above with `sjoin $JOBID $NODENAME` (e.g. `sjoin 456123 iris-123`)
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

       (node)$> cd ~/multiphysics-tutorial
       (node)$> cd inputs/qe
       (node)$> srun -n 1 pw.x < si.scf.efield2.in

We will see the calculation progress, this serial execution (we forced `srun` to only use a single task) should take around 2 minutes.

Next, we will clean up the directory holding output files, and re-run the example in parallel:

       (node)$> rm -rf out
       (node)$> srun pw.x < si.scf.efield2.in > si.scf.efield2.out

When the execution ends, we can take a look at the last 10 lines of output and check the execution time:

       (node)$> tail si.scf.efield2.out

You can now try to run the same example but testing some different things:

- using all cores in a single node, with one core per MPI process (28 tasks with 1 core per task)
- explicitly setting the number of OpenMP threads and running 1 MPI process and 28 threads (1 task with 28 cores per task)
- running across several nodes and e.g. compare execution on the _broadwell_ nodes vs _skylake_ based nodes (use `sbatch -C broadwell` to limit your jobs to nodes with this feature, idem for skylake)

Finally, we clean the environment by running `module purge`:

       (node)$> module purge
       (node)$> module list


### References

  - [QE: user's manual](http://www.quantum-espresso.org/resources/users-manual)
  - [QE: understanding parallelism](http://www.quantum-espresso.org/Doc/user_guide/node16.html)
  - [QE: running on parallel machines](http://www.quantum-espresso.org/Doc/user_guide/node17.html)
  - [QE: parallelization levels](http://www.quantum-espresso.org/Doc/user_guide/node18.html)


## OpenFOAM

Check for the available versions of OpenFOAM on Iris:

       (node)$> module avail openfoam

We will use the `cae/OpenFOAM/v1712-intel-2018a` version:

        (node)$> module load cae/OpenFOAM/v1712-intel-2018a

We load OpenFOAM's startup file:

       (node)$> source $FOAM_BASH

Now we will run the `reactingParcelFoam` solver of OpenFOAM on an example showing the spray-film cooling of hot boxes (lagrangian/reactingParcelFilmFoam/hotBoxes).
For reference, many examples are given in the installation directory of OpenFOAM, see `$FOAM_TUTORIALS`.

Before the main execution, some pre-processing steps:

       (node)$> cd ~/multiphysics-tutorial/inputs/openfoam
       (node)$> cp -rf 0.org 0
       (node)$> blockMesh
       (node)$> topoSet
       (node)$> subsetMesh c0 -patch wallFilm -overwrite
       (node)$> ./patchifyObstacles > log.patchifyObstacles 2>&1
       (node)$> extrudeToRegionMesh -overwrite
       (node)$> changeDictionary
       (node)$> rm -rf system/wallFilmRegion
       (node)$> cp -r system/wallFilmRegion.org system/wallFilmRegion
       (node)$> find ./0 -maxdepth 1 -type f -exec sed -i "s/wallFilm/\"(region0_to.*)\"/g" {} \;
       (node)$> paraFoam -touch
       (node)$> paraFoam -touch -region wallFilmRegion
       (node)$> decomposePar -region wallFilmRegion
       (node)$> decomposePar

Solver execution, note the environment variables we need to export:

       (node)$> srun --export MPI_BUFFER_SIZE,WM_PROJECT_DIR reactingParcelFilmFoam -parallel

Note that the solver execution will take a long time - you can interrupt and/or test in a larger job, which would require:

- editing the `decomposeParDict` file to change the `numberOfSubdomains` directive in order to match the new number of processes
- then rerun the `decomposePar` commands as above

Parenthesis: how can you achieve the best (fastest) execution time? Some questions to think about:

- is just increasing the number of cores optimal? (why not?)
- how can you increase processing speed on Iris?
- would you get an additional speedup if you compile OpenFOAM on the most recent architecture Iris nodes to take advantage of the newer instruction set available in their CPUs? (yes!)

After the main execution, post-processing steps:

       (node)$> reconstructPar -region wallFilmRegion
       (node)$> reconstructPar

You can now try to copy and run additional examples from OpenFOAM, note:

- the ones which include an `Allrun-parallel` file can be run in parallel
- you can run the `Allrun.pre` script to prepare the execution
- you have to run yourself further pre-execution instructions from the `Allrun-parallel` script
- instead of `runParallel $application 4` you will have to run mpirun with the correct parameters and the particular application name yourself
- last post-processing steps from `Allrun-parallel` have to be run manually

Finally, we clean the environment:

       (node)$> module purge
       (node)$> module list

### References

  - [OpenFOAM: user's guide](http://cfd.direct/openfoam/user-guide/)
  - [OpenFOAM: running applications in parallel](http://cfd.direct/openfoam/user-guide/running-applications-parallel/)


## ABINIT

Check for the available versions of ABINIT and load the latest:

       (node)$> module load abinit
       (node)$> module load chem/ABINIT


We will use one of ABINIT's parallel test cases to exemplify parallel execution.
For reference, many examples are given in the installation directory of ABINIT, see `$EBROOTABINIT/share/abinit-test`.

       (node)$> cd ~/multiphysics-tutorial/inputs/abinit
       (node)$> srun abinit < si_kpt_band_fft.files

After some initial processing and messages, we will see:

        finddistrproc.F90:394:WARNING
        Your input dataset does not let Abinit find an appropriate process distribution with nproc=    4
        Try to comment all the np* vars and set paral_kgb=    -4 to have advices on process distribution.

        abinit : WARNING -
         The product of npkpt, npfft, npband and npspinor is bigger than the number of processors.
         The user-defined values of npkpt, npfft, npband or npspinor will be modified,
         in order to bring this product below nproc .
         At present, only a very simple algorithm is used ...

        abinit : WARNING -
         Set npfft to 1

        initmpi_grid.F90:108:WARNING
          The number of band*FFT*kpt*spinor processors, npband*npfft*npkpt*npspinor should be
         equal to the total number of processors, nproc.
         However, npband   =    2           npfft    =    1           npkpt    =    1           npspinor =    1       and nproc    =    4

As shown above, ABINIT itself can give details into how to tune input parameters for the dataset used.

Edit the input file `si_kpt_band_fft.in` as per ABINIT's instructions, then re-run ABINIT.
The following message will be shown, with a list of parameters that you will need to edit in `si_kpt_band_fft`.

       "Computing all possible proc distrib for this input with nproc less than      4"

Next, ensure you can now run ABINIT on this example to completion.

Parenthesis: will a parallel application always allow execution on any number of cores? Some questions to think about:

- are there cases where an input problem cannot be split in some particular ways? (yes!)
- are all ways to split a problem optimal for solving it as fast as possible? (no!)
- is it possible to split a problem such that the solver has unbalanced cases and works much slower? (yes)
- is there a generic way to tune the problem in order to be solved as fast as possible? (no, it's domain & application specific!)

Finally, we clean the environment:

       (node)$> module purge
       (node)$> module list

### References

  - [ABINIT: user's guide](http://www.abinit.org/doc/helpfiles/for-v7.2/users/new_user_guide.html)
  - [ABINIT: tutorials](http://www.abinit.org/doc/helpfiles/for-v7.2/tutorial/welcome.html)



## NAMD

[NAMD](http://www.ks.uiuc.edu/Research/namd/), recipient of a 2002 Gordon Bell Award and a 2012 Sidney Fernbach Award, is a parallel molecular dynamics code designed for high-performance simulation of large biomolecular systems. Based on Charm++ parallel objects, NAMD scales to hundreds of cores for typical simulations and beyond 500,000 cores for the largest simulations.

The latest NAMD 2.13 is available on the `iris` cluster as of June 2019 in the development, and on Debian 8 nodes of `gaia` as of August 2017, let's check for it:

        (node)$> module avail namd
        (node)$> module load swenv/default-env/devel
        (node)$> module avail namd

        --------------------------- /opt/apps/resif/data/devel/default/modules/all ---------------------------
           chem/NAMD/2.13-foss-2019a-mpi

We will use one of the benchmark inputs of NAMD to test it, specifically the [reference](http://www.ks.uiuc.edu/Research/namd/utilities/)
_STMV (virus) benchmark (1,066,628 atoms, periodic, PME)_.


        (node)$> cd ~/multiphysics-tutorial/inputs/namd
        (node)$> tar xf stmv.tar.gz
        (node)$> cd stmv
        (node)$> module load chem/NAMD

Now, we will need to set the `outputName` parameter within the input file to the path that we want:

        (node)$> sed -i 's/^outputName.*$/outputName    generated-data/g' stmv.namd

Next we will perform the parallel execution of NAMD, showing its runtime output both on console and storing it to file using `tee`:

        (node)$> srun namd2 stmv.namd | tee out

## ASE

ASE is a Python library for working with atoms [*](https://wiki.fysik.dtu.dk/ase/_downloads/ase-talk.pdf).

ASE can interface with many external codes as `calculators`: Asap, GPAW, Hotbit, ABINIT, CP2K, CASTEP, DFTB+, ELK, EXCITING, FHI-aims, FLEUR, GAUSSIAN, Gromacs, Jacapo, LAMMPS, MOPAC, NWChem, SIESTA, TURBOMOLE and VASP. More details on the [official webpage](https://wiki.fysik.dtu.dk/ase/index.html).

Let us run the official short example _structure optimization of hydrogen molecule_ on the Iris cluster. Note that parallel executions of the external codes require specific environment variables to be set up, e.g. for NWChem it's `ASE_NWCHEM_COMMAND` which needs to include the `srun` parallel job launcher of Iris, which integrates with the MPI suites.

      (node)$> module avail NWChem ASE
      (node)$> module load chem/ASE/3.17.0-intel-2019a-Python-2.7.15 chem/NWChem/6.8.revision47-intel-2019a-Python-2.7.15
      (node)$> export ASE_NWCHEM_COMMAND='srun nwchem PREFIX.nw > PREFIX.out'
      (node)$> python
             >>> from ase import Atoms
             >>> from ase.optimize import BFGS
             >>> from ase.calculators.nwchem import NWChem
             >>> from ase.io import write
             >>> h2 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.7]])
             >>> h2.calc = NWChem(xc='PBE')
             >>> opt = BFGS(h2)
             >>> opt.run(fmax=0.02)
                   Step     Time          Energy         fmax
             BFGS:    0 11:55:55      -31.435218        2.2691
             BFGS:    1 11:55:55      -31.490762        0.3740
             BFGS:    2 11:55:56      -31.492780        0.0630
             BFGS:    3 11:55:57      -31.492837        0.0023
             True
             >>> write('H2.xyz', h2)
             >>> h2.get_potential_energy()
             -31.49283665375563

Note that the (very) important part here was to let ASE know how it can run NWChem in parallel, by explicitly setting the environment variable `ASE_NWCHEM_COMMAND` to the parallel execution commands for NWChem.

### References

  - [ASE: tutorials](https://wiki.fysik.dtu.dk/ase/tutorials/tutorials.html)
  - [ASE: calculators](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html)

## Now practice!

The main objective of this session was to get you accustomed to running scalable applications in parallel.

Remember that the benefit of running in a HPC/supercomputing environment comes only if your application can take advantage of parallel processing.

Now it's up to you to run your own test-cases, and discover how to optimize your executions on the UL HPC platform.
