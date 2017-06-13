-*- mode: markdown; mode: auto-fill; fill-column: 80 -*-

`README.md`

Copyright (c) 2015-2017 <Valentin.Plugaru@uni.lu> [ULHPC management team](mailto:<hpc-sysadmins@uni.lu>) [www](http://hpc.uni.lu)

-------------------

# UL HPC Tutorial: HPCG benchmarking on UL HPC platform

The objective of this tutorial is to compile and run one of the newest HPC benchmarks, [HPCG](http://www.hpcg-benchmark.org/), on top of the
[UL HPC](http://hpc.uni.lu) platform.  

You can work in groups for this training, yet individual work is encouraged to
ensure you understand and practice the usage of an HPC platform.  

In all cases, ensure you are able to [connect to the chaos and gaia cluster](https://hpc.uni.lu/users/docs/access.html).

	/!\ FOR ALL YOUR COMPILING BUSINESS, ENSURE YOU WORK ON A COMPUTING NODE

	(access)$> 	oarsub -I -l nodes=1,walltime=2

**Advanced users only**: rely on `screen` (see
  [tutorial](http://support.suso.com/supki/Screen_tutorial)) on the frontend
  prior to running any `oarsub` command to be more resilient to disconnection.  

The latest version of this tutorial is available on
[Github](https://github.com/ULHPC/tutorials/tree/devel/advanced/HPCG)

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
    $> wget https://software.sandia.gov/hpcg/downloads/hpcg-2.4.tar.gz
    $> tar xvzf hpcg-2.4.tar.gz
    $> cd hpcg-2.4     
    $> module avail MPI
    $> module load toolchain/ictce/7.3.5
    $> module list
    Currently Loaded Modules:
      1) compiler/icc/2015.3.187     3) toolchain/iccifort/2015.3.187            5) toolchain/iimpi/7.3.5                7) toolchain/ictce/7.3.5
      2) compiler/ifort/2015.3.187   4) mpi/impi/5.0.3.048-iccifort-2015.3.187   6) numlib/imkl/11.2.3.187-iimpi-7.3.50
	$> module show mpi/impi/5.0.3.048-iccifort-2015.3.187

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
    $> cat hpcg.
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

Re-compile HPCG with GNU C++, adapting the setup file  `Make.gcc` from `Make.Linux_MPI` to use the `mpicxx` wrapper and the GCC specific [architecture options](https://gcc.gnu.org/onlinedocs/gcc-4.9.2/gcc/i386-and-x86-64-Options.html).

    $> cd ~/TP
    $> make clean
    $> module purge
    $> module load mpi/OpenMPI/1.8.4-GCC-4.9.2
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
