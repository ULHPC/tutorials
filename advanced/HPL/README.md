-*- mode: markdown; mode: visual-line; fill-column: 80 -*-

Copyright (c) 2013-2017 UL HPC Team  <hpc-sysadmins@uni.lu>

        Time-stamp: <Mon 2017-06-12 13:51 svarrette>

------------------------------------------------------
# UL HPC Tutorial: High-Performance Linpack (HPL) benchmarking on UL HPC platform

[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/advanced/HPL/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/advanced/HPL/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)


The objective of this tutorial is to compile and run on of the reference HPC benchmarks, [HPL](http://www.netlib.org/benchmark/hpl/), on top of the [UL HPC](http://hpc.uni.lu) platform.

You can work in groups for this training, yet individual work is encouraged to ensure you understand and practice the usage of MPI programs on an HPC platform.
If not yet done, you should consider completing the [OSU Micro-benchmark](../OSU_MicroBenchmarks/) tutorial as it introduces the effective iusage of the different MPI suits available on the UL HPC platform.

In all cases, ensure you are able to [connect to the UL HPC  clusters](https://hpc.uni.lu/users/docs/access.html).


```bash
# /!\ FOR ALL YOUR COMPILING BUSINESS, ENSURE YOU WORK ON A (at least half) COMPUTING NODE
# Have an interactive job
(access)$> si -n 14                                      # iris
(access)$> srun -p interactive --qos qos-iteractive -n 14 --pty bash  # iris (long version)
(access)$> oarsub -I -l enclosure=1/nodes=1,walltime=4   # chaos / gaia
```


**Advanced users only**: rely on `screen` (see  [tutorial](http://support.suso.com/supki/Screen_tutorial) or the [UL HPC tutorial](https://hpc.uni.lu/users/docs/ScreenSessions.html) on the  frontend prior to running any `oarsub` or `srun/sbatch` command to be more resilient to disconnection.

The latest version of this tutorial is available on
[Github](https://github.com/ULHPC/tutorials/tree/devel/advanced/HPL)

## Objectives

[HPL](http://www.netlib.org/benchmark/hpl/) is a  portable implementation of the High-Performance Linpack (HPL) Benchmark for Distributed-Memory Computers. It is used as reference benchmark to provide data for the [Top500](http://top500.org) list and thus rank to supercomputers worldwide.
HPL rely on an efficient implementation of the Basic Linear Algebra Subprograms (BLAS). You have several choices at this level:

* Intel MKL
* [ATLAS](http://math-atlas.sourceforge.net/atlas_install/)
* [GotoBlas](https://www.tacc.utexas.edu/research-development/tacc-software/gotoblas2/)

The idea is to compare the different MPI and BLAS implementations available on the [UL HPC platform](http://hpc.uni.lu):

* [Intel MPI](http://software.intel.com/en-us/intel-mpi-library/) and the Intel MKL
* [OpenMPI](http://www.open-mpi.org/)
* [MVAPICH2](http://mvapich.cse.ohio-state.edu/overview/) (MPI-3 over OpenFabrics-IB, Omni-Path, OpenFabrics-iWARP, PSM, and TCP/IP)
* [ATLAS](http://math-atlas.sourceforge.net/atlas_install/)
* [GotoBlas](https://www.tacc.utexas.edu/research-development/tacc-software/gotoblas2/)

For the sake of time and simplicity, we will focus on the combination expected to lead to the best performant runs, _i.e._ Intel MKL and Intel MPI suite.


## Pre-requisites

On the **access** and a **computing** node of the cluster you're working on, clone the [ULHPC/tutorials](https://github.com/ULHPC/tutorials)  and [ULHPC/launcher-scripts](https://github.com/ULHPC/launcher-scripts) repositories

```bash
$> cd
$> mkdir -p git/ULHPC && cd  git/ULHPC
$> git clone https://github.com/ULHPC/launcher-scripts.git
$> git clone https://github.com/ULHPC/tutorials.git         # If not yet done
```

Prepare your working directory

```bash
$> mkdir -p ~/tutorials/HPL
$> cd ~/tutorials/HPL
$> ln -s ~/git/ULHPC/tutorials/advanced/HPL ref.ulhpc.d   # Keep a symlink to the reference tutorial
$> ln -s ref.ulhpc.d/Makefile .     # symlink to the root Makefile
```

Fetch and uncompress the latest version of the [HPL](http://www.netlib.org/benchmark/hpl/) benchmark (_i.e._ **version 2.2** at the time of writing).

```bash
$> cd ~/tutorials/HPL
$> mkdir src
$> cd src
# Download the latest version
$> export HPL_VERSION=2.2
$> wget --no-check-certificate http://www.netlib.org/benchmark/hpl/hpl-$(HPL_VERSION).tar.gz
$> tar xvzf hpl-$(HPL_VERSION).tar.gz
$> cd  hpl-$(HPL_VERSION)
```

## Building the HPL benchmark

We are first going to use the [Intel Cluster Toolkit Compiler Edition](http://software.intel.com/en-us/intel-cluster-toolkit-compiler/), which provides Intel C/C++ and Fortran compilers, Intel MPI.

```bash
$> cd ~/tutorials/HPL
# Load the appropriate module
$> module spider MPI     # Search for available modules featuring MPI
$> module load toolchain/intel   # On iris -- use 'module load toolchain/ictce' otherwise
$> module list
Currently Loaded Modules:
  1) compiler/GCCcore/6.3.0                   4) compiler/ifort/2017.1.132-GCC-6.3.0-2.27                 7) toolchain/iimpi/2017a
  2) tools/binutils/2.27-GCCcore-6.3.0        5) toolchain/iccifort/2017.1.132-GCC-6.3.0-2.27             8) numlib/imkl/2017.1.132-iimpi-2017a
  3) compiler/icc/2017.1.132-GCC-6.3.0-2.27   6) mpi/impi/2017.1.132-iccifort-2017.1.132-GCC-6.3.0-2.27   9) toolchain/intel/2017a
```

You notice that Intel MKL is now loaded.

Read the `INSTALL` file. In particular, you'll have to edit and adapt a new makefile `Make.intel64`
(inspired from `setup/Make.Linux_PII_CBLAS` typically)

```bash
$> cd ~/tutorials/HPL/src/hpl-2.2
$> cp setup/Make.Linux_Intel64 Make.intel64
```

Once tweaked, run the compilation by:

```bash
$> make arch=intel64 clean_arch_all
$> make arch=intel64
```

But **first**, you will need to configure correctly the file `Make.intel64`.
Take your favorite editor (`vim`, `nano`, etc.) to modify it. In particular, you should adapt:

* `TOPdir` to point to the directory holding the HPL sources (_i.e._ where you uncompress them: ` $(HOME)/tutorials/HPL/src/hpl-2.2`)
* Adapt the `MP*` variables to point to the appropriate MPI libraries path.
* (eventually) adapt the `CCFLAGS`

Here is for instance a suggested difference for intel MPI:

```diff
--- setup/Make.Linux_Intel64    2016-02-24 02:10:50.000000000 +0100
+++ Make.intel64        2017-06-12 13:48:31.016524323 +0200
@@ -61,13 +61,13 @@
 # - Platform identifier ------------------------------------------------
 # ----------------------------------------------------------------------
 #
-ARCH         = Linux_Intel64
+ARCH         = $(arch)
 #
 # ----------------------------------------------------------------------
 # - HPL Directory Structure / HPL library ------------------------------
 # ----------------------------------------------------------------------
 #
-TOPdir       = $(HOME)/hpl
+TOPdir       = $(HOME)/tutorials/HPL/src/hpl-2.2
 INCdir       = $(TOPdir)/include
 BINdir       = $(TOPdir)/bin/$(ARCH)
 LIBdir       = $(TOPdir)/lib/$(ARCH)
@@ -81,9 +81,9 @@
 # header files,  MPlib  is defined  to be the name of  the library to be
 # used. The variable MPdir is only used for defining MPinc and MPlib.
 #
-# MPdir        = /opt/intel/mpi/4.1.0
-# MPinc        = -I$(MPdir)/include64
-# MPlib        = $(MPdir)/lib64/libmpi.a
+MPdir        = $(I_MPI_ROOT)/intel64
+MPinc        = -I$(MPdir)/include
+MPlib        = $(MPdir)/lib/libmpi.a
 #
 # ----------------------------------------------------------------------
 # - Linear Algebra library (BLAS or VSIPL) -----------------------------
@@ -178,7 +178,7 @@
 CC       = mpiicc
 CCNOOPT  = $(HPL_DEFS)
 OMP_DEFS = -openmp
-CCFLAGS  = $(HPL_DEFS) -O3 -w -ansi-alias -i-static -z noexecstack -z relro -z now -nocompchk -Wall
+CCFLAGS  = $(HPL_DEFS) -O3 -w -ansi-alias -i-static -z noexecstack -z relro -z now -nocompchk -Wall -xHost
 #
 # On some platforms,  it is necessary  to use the Fortran linker to find
 # the Fortran internals used in the BLAS library.
```






-----------
# OLD notes



The objective of this practical session is to compare the performances of HPL runs compiled under different combination:

1. HPL + Intel MKL + Intel MPI
2. HPL + GCC + GotoBLAS2 + Open MPI

The benchamrking campain will typically involves successively:

* a single node
* two nodes, ideally belonging to the same enclosure
* two nodes, belonging to different enclosures

## Runs on a single node

### High-Performance Linpack (HPL) with Intel Suite

We are first going to use the
[Intel Cluster Toolkit Compiler Edition](http://software.intel.com/en-us/intel-cluster-toolkit-compiler/),
which provides Intel C/C++ and Fortran compilers, Intel MPI & Intel MKL.

Resources:

* [HPL](http://www.netlib.org/benchmark/hpl/)

Get the latest release:

    $> mkdir ~/TP && cd ~/TP
    $> wget http://www.netlib.org/benchmark/hpl/hpl-2.1.tar.gz
    $> tar xvzf hpl-2.1.tar.gz
    $> cd hpl-2.1
    $> module avail MPI
    $> module load toolchain/ictce/7.3.5
    $> module list
    Currently Loaded Modules:
      1) compiler/icc/2015.3.187     3) toolchain/iccifort/2015.3.187            5) toolchain/iimpi/7.3.5                7) toolchain/ictce/7.3.5
      2) compiler/ifort/2015.3.187   4) mpi/impi/5.0.3.048-iccifort-2015.3.187   6) numlib/imkl/11.2.3.187-iimpi-7.3.50
	$> module show mpi/impi/5.0.3.048-iccifort-2015.3.187
	$> module show numlib/imkl/11.2.3.187-iimpi-7.3.5

You notice that Intel MKL is now loaded.

Read the `INSTALL` file.

In particular, you'll have to edit and adapt a new makefile `Make.intel64`
(inspired from `setup/Make.Linux_PII_CBLAS` typically) and run the compilation
by

	$> make arch=intel64 clean_arch_all
	$> make arch=intel64

Some hint to succeed:

* rely on `mpiicc` as compiler and linker
* rely on `$(I_MPI_ROOT)` (see `module show impi`) for the `MPdir` entry
* similarly, use `$(MKLROOT)` for the `LAdir` entry (see `module show imkl`)
* Effective use of MKL (in particular):

		LAdir        = $(MKLROOT)
		HPLlibHybrid = $(LAdir)/benchmarks/mp_linpack/lib_hybrid/intel64/libhpl_hybrid.a
		LAinc        = -I$(LAdir)/include
		LAlib        = -L$(LAdir)/lib/intel64 -Wl,--start-group $(LAdir)/lib/intel64/libmkl_intel_lp64.a $(LAdir)/lib/intel64/libmkl_intel_thread.a $(LAdir)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -ldl $(HPLlibHybrid)

If you don't succeed by yourself, use the following makefile:

    wget https://raw.githubusercontent.com/ULHPC/tutorials/devel/advanced/HPL/src/hpl-2.1/Make.intel64

Once compiled, ensure you are able to run it:

	$> cd bin/intel64
	$> cat HPL.dat
	$> mpirun -hostfile $OAR_NODEFILE ./xhpl

Now you'll have to find the optimal set of parameters for using a single
node. You can use the following site:
[HPL Calculator](http://hpl-calculator.sourceforge.net/) to find good parameters
and expected performances and adapt `bin/intel64/HPL.dat` accordingly.


### HPL with GCC and GotoBLAS2 and Open MPI

Another alternative is to rely on [GotoBlas](https://www.tacc.utexas.edu/research-development/tacc-software/gotoblas2/).

Get the sources and compile them:

     # A copy of `GotoBLAS2-1.13.tar.gz` is available in `/tmp` on the access nodes of the cluster
     $> cd ~/TP
     $> module purge
     $> module load mpi/OpenMPI/1.8.4-GCC-4.9.2
     $> tar xvzf /tmp/GotoBLAS2-1.13.tar.gz
     $> mv GotoBLAS2 GotoBLAS2-1.13
     $> cd GotoBLAS2-1.13
     $> make BINARY=64 TARGET=NEHALEM NUM_THREADS=1
     [...]
      GotoBLAS build complete.

        OS               ... Linux
        Architecture     ... x86_64
        BINARY           ... 64bit
        C compiler       ... GCC  (command line : gcc)
        Fortran compiler ... GFORTRAN  (command line : gfortran)
        Library Name     ... libgoto2_nehalemp-r1.13.a (Multi threaded; Max num-threads is 12)


If you don't use `TARGET=NEHALEM`, you'll encounter the error mentionned
[here](http://saintaardvarkthecarpeted.com/blog/archive/2011/05/Trouble_compiling_GotoBLAS2_on_newer_CPU.html))

Now you can restart HPL compilation by creating (and adapting) a
`Make.gotoblas2` and running the compilation by:

	$> make arch=gotoblas2

Once compiled, ensure you are able to run it:

	$> cd bin/gotoblas2
	$> cat HPL.dat
	$> mpirun -x LD_LIBRARY_PATH -hostfile $OAR_NODEFILE ./xhpl


## Benchmarking on two nodes

Restart the benchmarking campain (in the three cases) in the following context:

* 2 nodes belonging to the same enclosure. Use for that:

		$> oarsub -l enclosure=1/nodes=2,walltime=8 […]

* 2 nodes belonging to the different enclosures:

		$> oarsub -l enclosure=2/core=1,walltime=8 […]

## Now for Lazy / frustrated persons

You will find in the [UL HPC tutorial](https://github.com/ULHPC/tutorials)
repository, under the `advanced/HPL` directory, a set of tools / script that
facilitate the running and analysis of this tutorial that you can use/adapt to
suit your needs.

In particular, once in the `advanced/HPL` directory:

* running `make fetch` will automatically download the archives for HPL,
  GotoBLAS2 and ATLAS (press enter at the end of the last download)
* some examples of working `Makefile` for HPL used in sample experiments are
  proposed in `src/hpl-2.1`
* a launcher script is proposed in `runs/launch_hpl_bench`. This script was used
  to collect some sample runs for the three experiments mentionned in this
  tutorial as follows:

        # Run for HPL with iMKL (icc + iMPI)
		(access-chaos)$> oarsub -S "./launch_hpl_bench --module ictce --arch intel64 --serious"

		# Run for HPL with GotoBLAS2 (gcc + OpenMPI)
        (access-chaos)$> oarsub -S "./launch_hpl_bench --module OpenMPI --arch gotoblas2 --serious"

		# Run for HPL with ATLAS (gcc + MVAPICH2)
        (access-chaos)$> oarsub -S "./launch_hpl_bench --module MVAPICH2 --arch atlas --serious"

  In particular, the `runs/data/` directory host the results of these runs on 2
  nodes belonging to the same enclosure

* run `make plot` to invoke the [Gnuplot](http://www.gnuplot.info/) script
  `plots/benchmark_HPL.gnuplot` and generate various plots from the sample
  runs.

In particular, you'll probably want to see the comparison figure extracted from
the sample run `plots/benchmark_HPL_2H.pdf`

A PNG version of this plot is available on
[Github](https://raw.github.com/ULHPC/tutorials/devel/advanced/HPL/plots/benchmark_HPL_2H.png)

![HPL run on 2 hosts](https://raw.github.com/ULHPC/tutorials/devel/advanced/HPL/plots/benchmark_HPL_2H.png)
