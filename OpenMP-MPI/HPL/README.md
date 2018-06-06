[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/advanced/OSU_MicroBenchmarks/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/advanced/HPL/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/advanced/HPL/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# High-Performance Linpack (HPL) benchmarking on UL HPC platform

     Copyright (c) 2013-2017 UL HPC Team  <hpc-sysadmins@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/advanced/OSU_MicroBenchmarks/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/advanced/OSU_MicroBenchmarks/slides.pdf)

The objective of this tutorial is to compile and run on of the reference HPC benchmarks, [HPL](http://www.netlib.org/benchmark/hpl/), on top of the [UL HPC](http://hpc.uni.lu) platform.

You can work in groups for this training, yet individual work is encouraged to ensure you understand and practice the usage of MPI programs on an HPC platform.
If not yet done, you should consider completing the [OSU Micro-benchmark](../OSU_MicroBenchmarks/) tutorial as it introduces the effective usage of the different MPI suits available on the UL HPC platform.

In all cases, ensure you are able to [connect to the UL HPC  clusters](https://hpc.uni.lu/users/docs/access.html).


```bash
# /!\ FOR ALL YOUR COMPILING BUSINESS, ENSURE YOU WORK ON A (at least half) COMPUTING NODE
# Have an interactive job
(access)$> si -n 14                                      # iris
(access)$> srun -p interactive --qos qos-iteractive -n 14 --pty bash  # iris (long version)
(access)$> oarsub -I -l enclosure=1/nodes=1,walltime=4   # chaos / gaia
```


**Advanced users only**: rely on `screen` (see  [tutorial](http://support.suso.com/supki/Screen_tutorial) or the [UL HPC tutorial](https://hpc.uni.lu/users/docs/ScreenSessions.html) on the  frontend prior to running any `oarsub` or `srun/sbatch` command to be more resilient to disconnection.

The latest version of this tutorial is available on [Github](https://github.com/ULHPC/tutorials/tree/devel/advanced/HPL)
Finally, advanced MPI users might be interested to take a look at the [Intel Math Kernel Library Link Line Advisor](https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor).

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
$> wget --no-check-certificate http://www.netlib.org/benchmark/hpl/hpl-${HPL_VERSION}.tar.gz
$> tar xvzf hpl-${HPL_VERSION}.tar.gz
$> cd  hpl-${HPL_VERSION}
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

If you don't succeed by yourself, use the following [makefile](https://raw.githubusercontent.com/ULHPC/tutorials/devel/advanced/HPL/src/hpl-2.2/Make.intel64):

```
$> cd ~/tutorials/HPL
$> cp ref.ulhpc.d/src/hpl-2.2/Make.intel64 src/hpl-2.2/Make.intel64
```

Once compiled, ensure you are able to run it:

```bash
$> cd ~/tutorials/HPL/src/hpl-2.2/bin/intel64
$> cat HPL.dat      # Default (dummy) HPL.dat  input file

# On Slurm cluster (iris)
$> srun -n $SLURM_NTASKS ./xhpl

# On OAR clusters (gaia, chaos)
$> mpirun -hostfile $OAR_NODEFILE ./xhpl
```

### Preparing batch runs

We are now going to prepare launcher scripts to permit passive runs (typically in the `{default | batch}` queue).
We will place them in a separate directory (`runs/`) as it will host the outcomes of the executions on the UL HPC platform .

```bash
$> cd ~/tutorials/HPL
$> mkdir runs    # Prepare the specific run directory
```

Now you'll have to find the optimal set of parameters for using a single
node. You can use the following site:
[HPL Calculator](http://hpl-calculator.sourceforge.net/) to find good parameters
and expected performances and adapt `bin/intel64/HPL.dat` accordingly.
Here we are going to use reasonable choices as outline from [this website](http://www.advancedclustering.com/act_kb/tune-hpl-dat-file/)


### Slurm launcher (Intel MPI)

Copy and adapt the [default SLURM launcher](https://github.com/ULHPC/launcher-scripts/blob/devel/slurm/launcher.default.sh) you should have a copy in `~/git/ULHPC/launcher-scripts/slurm/launcher.default.sh`

```bash
$> cd ~/tutorials/HPL/runs
# Prepare a laucnher for intel suit
$> cp ~/git/ULHPC/launcher-scripts/slurm/launcher.default.sh launcher-HPL.intel.sh
```

Take your favorite editor (`vim`, `nano`, etc.) to modify it according to your needs.

Here is for instance a suggested difference for intel MPI:

```diff
--- ~/git/ULHPC/launcher-scripts/slurm/launcher.default.sh  2017-06-11 23:40:34.007152000 +0200
+++ launcher-HPL.intel.sh       2017-06-11 23:41:57.597055000 +0200
@@ -10,8 +10,8 @@
 #
 #          Set number of resources
 #
-#SBATCH -N 1
+#SBATCH -N 2
 #SBATCH --ntasks-per-node=28
 ### -c, --cpus-per-task=<ncpus>
 ###     (multithreading) Request that ncpus be allocated per process
 #SBATCH -c 1
@@ -64,15 +64,15 @@
 module load toolchain/intel

 # Directory holding your built applications
-APPDIR="$HOME"
+APPDIR="$HOME/tutorials/HPL/src/hpl-2.2/bin/intel64"
 # The task to be executed i.E. your favorite Java/C/C++/Ruby/Perl/Python/R/whatever program
 # to be invoked in parallel
-TASK="${APPDIR}/app.exe"
+TASK="${APPDIR}/xhpl"

 # The command to run
-CMD="${TASK}"
+# CMD="${TASK}"
 ### General MPI Case:
-# CMD="srun -n $SLURM_NTASKS ${TASK}"
+CMD="srun -n $SLURM_NTASKS ${TASK}"
 ### OpenMPI case if you wish to specialize the MCA parameters
 #CMD="mpirun -np $SLURM_NTASKS --mca btl openib,self,sm ${TASK}"
```

Now you should create an input `HPL.dat` file within the `runs/`.

```bash
$> cd ~/tutorials/HPL/runs
$> cp ../ref.ulhpc.d/HPL.dat .
$> ll
total 0
-rw-r--r--. 1 svarrette clusterusers 1.5K Jun 12 15:38 HPL.dat
-rwxr-xr-x. 1 svarrette clusterusers 2.7K Jun 12 15:25 launcher-HPL.intel.sh
```

You are ready for testing a batch job:


```bash
$> cd ~/tutorials/HPL/runs
$> sbatch ./launcher-HPL.intel.sh
$> sq     # OR (long version) squeue -u $USER
```

**(bonus)** Connect to one of the allocated nodes and run `htop` (followed by `u` to select process run under your username, and `F5` to enable the tree-view.

Now you can check the output of the HPL runs:

```bash
$> grep WR slurm-<jobid>.out    # /!\ ADAPT <jobid> appropriately.
```

Of course, we made here a small test and optimizing the HPL parameters to get the best performances and efficiency out of a given HPC platform is not easy.
Below are some plots obtained when benchmarking the `iris` cluster and seeking the best set of parameters across increasing number of nodes (see [this blog post](https://hpc.uni.lu/blog/2017/preliminary-performance-results-of-the-iris-cluster/))

![](https://hpc.uni.lu/images/benchs/benchmark_HPL-iris_25N.png)
![](https://hpc.uni.lu/images/benchs/benchmark_HPL-iris_50N.png)
![](https://hpc.uni.lu/images/benchs/benchmark_HPL-iris_75N.png)
![](https://hpc.uni.lu/images/benchs/benchmark_HPL-iris_100N.png)
