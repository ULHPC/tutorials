-*- mode: markdown; mode: visual-line; fill-column: 80 -*-

Author: Valentin Plugaru <Valentin.Plugaru@uni.lu>  
Copyright (c) 2015-2017 UL HPC Team  <hpc-sysadmins@uni.lu>

------------------------------------------------------
# UL HPC MPI Tutorial: High Performance Conjugate Gradients (HPCG) benchmarking on UL HPC platform

[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/advanced/HPCG/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/advanced/HPCG/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)


The objective of this tutorial is to compile and run one of the newest HPC benchmarks, [High Performance Conjugate Gradients (HPCG)](http://www.hpcg-benchmark.org/), on top of the
[UL HPC](http://hpc.uni.lu) platform.

You can work in groups for this training, yet individual work is encouraged to ensure you understand and practice the usage of MPI programs on an HPC platform.
If not yet done, you should consider completing the [OSU Micro-benchmark](../OSU_MicroBenchmarks/) and [HPL](../HPL/) tutorials.

In all cases, ensure you are able to [connect to the UL HPC  clusters](https://hpc.uni.lu/users/docs/access.html).


```bash
# /!\ FOR ALL YOUR COMPILING BUSINESS, ENSURE YOU WORK ON A (at least half) COMPUTING NODE
# Have an interactive job
(access)$> si -n 14                                      # iris
(access)$> srun -p interactive --qos qos-iteractive -n 14 --pty bash  # iris (long version)
(access)$> oarsub -I -l enclosure=1/nodes=1,walltime=4   # chaos / gaia
```

**Advanced users only**: rely on `screen` (see  [tutorial](http://support.suso.com/supki/Screen_tutorial) or the [UL HPC tutorial](https://hpc.uni.lu/users/docs/ScreenSessions.html) on the  frontend prior to running any `oarsub` or `srun/sbatch` command to be more resilient to disconnection.

The latest version of this tutorial is available on [Github](https://github.com/ULHPC/tutorials/tree/devel/advanced/HPCG).
Finally, advanced MPI users might be interested to take a look at the [Intel Math Kernel Library Link Line Advisor](https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor).

## Objectives

The High Performance Conjugate Gradient [HPCG](http://www.hpcg-benchmark.org/) project is an effort to create a more relevant metric for ranking HPC systems than the High Performance LINPACK (HPL) benchmark, which is currently used in
the [Top500](http://top500.org) ranking.

HPCG exhibits the following patterns:
* Dense and sparse computations
* Dense and sparse collective operations
* Data-driven parallelism (unstructured sparse triangular solves)

For more details, check out:
* [Toward a New Metric for Ranking High Performance Computing Systems](https://software.sandia.gov/hpcg/doc/HPCG-Benchmark.pdf)
* [Technical specification](https://software.sandia.gov/hpcg/doc/HPCG-Specification.pdf)

HPCG is written in C++, with OpenMP and MPI parallelization capabilities, thus requires a C++ compiler with OpenMP support, and/or a MPI library.

The objective of this practical session is to compare the performance obtained by running HPCG
compiled with different compilers and options:

1. HPCG + Intel C++ + Intel MPI
    - architecture native build, using the most recent supported instruction set (AVX2/FMA3)
    - SSE4.1 instruction set build
2. HPCG + GNU C++ + Open MPI
    - architecture native build, using the most recent supported instruction set (AVX2/FMA3)
    - SSE4.1 instruction set build

The benchmarking tests should be performed on:

* a single node
* two nodes, ideally belonging to the same enclosure
* two nodes, belonging to different enclosures

## Executions on a single node

### High Performance Conjugate Gradient (HPCG) with the Intel Suite

We are first going to use the
[Intel Cluster Toolkit Compiler Edition](http://software.intel.com/en-us/intel-cluster-toolkit-compiler/),
which provides Intel C/C++ and Fortran compilers, Intel MPI & Intel MKL.

Resources:

* [HPCG project](http://hpcg-benchmark.org/)

Get the latest release:

    $> mkdir ~/TP && cd ~/TP
    $> wget http://www.hpcg-benchmark.org/downloads/hpcg-3.0.tar.gz
    $> tar xvzf hpcg-3.0.tar.gz
    $> cd hpcg-3.0
    $> module avail MPI
    $> module load toolchain/intel
    $> module list
    Currently Loaded Modules:
      1) compiler/GCCcore/6.3.0                   4) compiler/ifort/2017.1.132-GCC-6.3.0-2.27                 7) toolchain/iimpi/2017a
      2) tools/binutils/2.27-GCCcore-6.3.0        5) toolchain/iccifort/2017.1.132-GCC-6.3.0-2.27             8) numlib/imkl/2017.1.132-iimpi-2017a
      3) compiler/icc/2017.1.132-GCC-6.3.0-2.27   6) mpi/impi/2017.1.132-iccifort-2017.1.132-GCC-6.3.0-2.27   9) toolchain/intel/2017a
    $> module show mpi/impi/2017.1.132-iccifort-2017.1.132-GCC-6.3.0-2.27

Read the `INSTALL` file.

In particular, you'll have to edit a new makefile `Make.intel64`
(inspired from `setup/Make.MPI_ICPC` typically), adapting:

1. the CXX variable specifying the C++ compiler (use `mpiicpc` for the MPI Intel C++ wrapper)
2. the CXXFLAGS variable
with architecture-specific compilation flags (see [this Intel article](https://software.intel.com/en-us/articles/performance-tools-for-software-developers-intel-compiler-options-for-sse-generation-and-processor-specific-optimizations))

Once the configuration file is prepared, run the compilation with:
	$> make arch=intel64

Once compiled, ensure that you are able to run it:

    $> cd bin
    $> cat hpcg.dat
    $> mkdir intel64-optimized
    $> mv xhpcg intel64-optimized
    $> cd intel64-optimized
    $> ln -s ../hpcg.dat .
    $> mpirun -hostfile $OAR_NODEFILE ./xhpcg

As configured in the default `hpcg.dat`, HPCG generates a synthetic discretized three-dimensional partial differential equation model problem with Nx=Ny=Nz=104 local subgrid dimensions. NPx, NPy, NPz are a factoring of the MPI process space, giving a global domain dimension of (Nx * NPx ) * (Ny * NPy ) * (Nz * NPz).

You can tune Nx, Ny, Nz to increase/decrease the problem size, yet take care not to generate a problem whose local node grid representation exceeds computing node memory.

The result of your experiments will be stored in the directory HPCG was started in, in a `HPCG-Benchmark-2.4_$(date).yaml` file. Check out the benchmark result (GFLOP/s) in the final summary section:

    $> grep "HPCG result is" $file.yaml

In addition to the architecture optimized build, re-generate xhpcg to with the compiler options to support only the SSE4.1 instruction set (common across all UL HPC computing nodes) and perform the same experiment, in a new `intel64-generic` directory.

### HPCG with GNU C++ and Open MPI

Re-compile HPCG with GNU C++, adapting the setup file  `Make.gcc` from `Make.Linux_MPI` to use the `mpicxx` wrapper and the GCC specific [architecture options](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html).

    $> cd ~/TP
    $> make clean
    $> module purge
    $> module load mpi/OpenMPI
    $> make arch=gcc

Once compiled, ensure you are able to run it:

    $> cd bin
    $> cat hpcg.dat
    $> mkdir gnu-optimized
    $> mv xhpcg gnu-optimized
    $> cd gnu-optimized
    $> ln -s ../hpcg.dat .
    $> mpirun -x PATH -x LD_LIBRARY_PATH -hostfile $OAR_NODEFILE ./xhpcg


## Benchmarking on two nodes

Restart the benchmarking campaign (for both the Intel and GCC) in the following context:

* 2 nodes belonging to the same enclosure. Use for that:

    $> oarsub -l enclosure=1/nodes=2,walltime=1 […]

* 2 nodes belonging to the different enclosures:

    $> oarsub -l enclosure=2/core=1,walltime=1 […]

## Benchmarking with OpenMP active

Finally, activate OpenMP support when building HPCG, adapting for the Intel and GCC suites `Make.MPI_ICPC_OMP` and `Make.MPI_GCC_OMP` respectively.
As before, perform single and multiple node benchmarks.

How is the performance result for the OpenMP+MPI vs MPI-only executions?
