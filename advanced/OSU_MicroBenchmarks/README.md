-*- mode: markdown; mode: auto-fill; fill-column: 80 -*-

`README.md`

Copyright (c) 2013 Sebastien Varrette <Sebastien.Varrette@uni.lu>

        Time-stamp: <Dim 2013-11-11 19:21 svarrette>

-------------------


# UL HPC Tutorial: OSU Micro-Benchmarks on UL HPC platform

The objective of this tutorial is to compile and run on of the [OSU micro-benchmarks](http://mvapich.cse.ohio-state.edu/benchmarks/) which permit to measure the performance of an MPI implementation.  

You can work in groups for this training, yet individual work is encouraged to
ensure you understand and practice the usage of an HPC platform.  

In all cases, ensure you are able to [connect to the chaos and gaia cluster](https://hpc.uni.lu/users/docs/access.html). 

	/!\ FOR ALL YOUR COMPILING BUSINESS, ENSURE YOU WORK ON A COMPUTING NODE
	
	(access)$> 	oarsub -I -l nodes=1,walltime=4

**Advanced users only**: rely on `screen` (see
  [tutorial](http://support.suso.com/supki/Screen_tutorial)) on the frontend
  prior to running any `oarsub` command to be more resilient to disconnection.  

The latest version of this tutorial is available on
[Github](https://github.com/ULHPC/tutorials/tree/devel/advanced/OSU_MicroBenchmarks)

## Objectives

The [OSU micro-benchmarks](http://mvapich.cse.ohio-state.edu/benchmarks/) features a serie of MPI benchmarks to measure the performances of various MPI operations: 

* __Point-to-Point MPI Benchmarks__: Latency, multi-threaded latency, multi-pair latency, multiple bandwidth / message rate test bandwidth, bidirectional bandwidth
* __Collective MPI Benchmarks__: Collective latency tests for various MPI collective operations such as MPI_Allgather, MPI_Alltoall, MPI_Allreduce, MPI_Barrier, MPI_Bcast, MPI_Gather, MPI_Reduce, MPI_Reduce_Scatter, MPI_Scatter and vector collectives.
* __One-sided MPI Benchmarks__: one-sided put latency (active/passive), one-sided put bandwidth (active/passive), one-sided put bidirectional bandwidth, one-sided get latency (active/passive), one-sided get bandwidth (active/passive), one-sided accumulate latency (active/passive), compare and swap latency (passive), and fetch and operate (passive) for MVAPICH2 (MPI-2 and MPI-3).

The latest version (4.2 at the time of writting) also features [OpenSHMEM]() benchmarks, a 1-sided communications library.

In this tutorial, we will focus on two of the available tests:

* `osu_latency` - Latency Test

  The latency tests are carried out in a ping-pong fashion. The sender sends a message with a certain data size to the receiver and waits for a reply from the receiver. The receiver receives the message from the sender and sends back a reply with the same data size. Many iterations of this ping-pong test are carried out and average one-way latency numbers are obtained. Blocking version of MPI functions (MPI_Send and MPI_Recv) are used in the tests.
* `osu_bw` - Bandwidth Test

  The bandwidth tests were carried out by having the sender sending out a fixed number (equal to the window size) of back-to-back messages to the receiver and then waiting for a reply from the receiver. The receiver sends the reply only after receiving all these messages. This process is repeated for several iterations and the bandwidth is calculated based on the elapsed time (from the time sender sends the first message until the time it receives the reply back from the receiver) and the number of bytes sent by the sender. The objective of this bandwidth test is to determine the maximum sustained date rate that can be achieved at the network level. Thus, non-blocking version of MPI functions (MPI_Isend and MPI_Irecv) were used in the test.

The idea is to compare the different MPI implementations available on the [UL HPC platform](http://hpc.uni.lu).: 

* [Intel MPI](http://software.intel.com/en-us/intel-mpi-library/)
* [OpenMPI](http://www.open-mpi.org/)
* [MVAPICH2](http://mvapich.cse.ohio-state.edu/overview/mvapich2/)

The benchamrking campain will typically involves for each MPI suit: 

* two nodes, belonging to the same enclosure
* two nodes, belonging to different enclosures

## Pre-requisites

Clone the [launcher-script repository](https://github.com/ULHPC/launcher-scripts)

	$> cd 
	$> mkdir -p git/ULHPC && cd  git/ULHPC
	$> git clone https://github.com/ULHPC/launcher-scripts.git
	
Now you shall get the latest release of the [OSU micro-benchmarks](http://mvapich.cse.ohio-state.edu/benchmarks/) (4.2 at the moment of writing)

	$> mkdir ~/TP && cd ~/TP
    $> wget http://mvapich.cse.ohio-state.edu/benchmarks/osu-micro-benchmarks-4.2.tar.gz
    $> tar xvzf osu-micro-benchmarks-4.2.tar.gz
    $> cd osu-micro-benchmarks-4.2

There are two tests, `osu_cas_flush` and `osu_fop_flush`, which uses some MPI primitives defined in the MPI 3.0 standard, thus unavailable in OpenMPI and iMPI on the cluster. So we will have to patch the sources to prevent the generation of these two benchamrks:

	$> cd ~/TP/osu-micro-benchmarks-4.2
	$> cd mpi/one-sided
	
Now create a file `Makefile.am.patch` with the following content: 

	$> cat Makefile.am.patch
	--- Makefile.am.old     2013-11-12 09:44:11.916456641 +0100
	+++ Makefile.am 2013-11-12 09:44:24.363424661 +0100
	@@ -1,2 +1,2 @@
	 one_sideddir = $(pkglibexecdir)/mpi/one-sided
	-one_sided_PROGRAMS = osu_acc_latency osu_passive_acc_latency osu_get_bw osu_get_latency osu_put_bibw osu_put_bw osu_put_latency osu_passive_get_latency osu_passive_get_bw osu_passive_put_latency osu_passive_put_bw osu_cas_flush osu_fop_flush
	+one_sided_PROGRAMS = osu_acc_latency osu_passive_acc_latency osu_get_bw osu_get_latency osu_put_bibw osu_put_bw osu_put_latency osu_passive_get_latency osu_passive_get_bw osu_passive_put_latency osu_passive_put_bw
	
… And apply the patch as follows:

	$> patch -p0 < Makefile.am.patch
	patching file Makefile.am
	
And update the Autotools chain configuration :

	$> cd ~/TP/osu-micro-benchmarks-4.2
	$> autoreconf && automake


## OSU Micro-benchmarks with Intel MPI

We are first going to use the
[Intel Cluster Toolkit Compiler Edition](http://software.intel.com/en-us/intel-cluster-toolkit-compiler/),
which provides Intel C/C++ and Fortran compilers, Intel MPI. 

Get the latest release: 

    $> mkdir ~/TP && cd ~/TP
    $> wget http://mvapich.cse.ohio-state.edu/benchmarks/osu-micro-benchmarks-4.2.tar.gz
    $> tar xvzf osu-micro-benchmarks-4.2.tar.gz
    $> cd osu-micro-benchmarks-4.2
    $> module avail 2>&1 | grep -i MPI
    $> module load ictce
    $> module list
	Currently Loaded Modulefiles:
			1) icc/2013.5.192     2) ifort/2013.5.192   3) impi/4.1.1.036     4) imkl/11.0.5.192    5) ictce/5.5.0
    $> mkdir build.impi && cd build.impi
    $> ../configure CC=mpiicc --prefix=`pwd`/install
	$> make && make install 

Once compiled, ensure you are able to run it: 

	$> cd install/libexec/osu-micro-benchmarks/mpi/one-sided/
	$> mpirun -hostfile $OAR_NODEFILE -perhost 1 ./osu_get_latency

Now you can use the [MPI generic launcher](https://github.com/ULHPC/launcher-scripts/blob/devel/bash/MPI/mpi_launcher.sh) to run the code: 

	$> cd ~/TP/osu-micro-benchmarks-4.2/
	$> mkdir runs.impi && cd runs.impi
	$> ln -s 




### HPL with GCC and GotoBLAS2 and Open MPI

Another alternative is to rely on [GotoBlas](http://www.tacc.utexas.edu/tacc-projects/gotoblas2/downloads/).  

Get the sources and compile them: 

     # A copy of `GotoBLAS2-1.13.tar.gz` is available in `/tmp` on the access nodes of the cluster
     $> cd ~/TP
     $> module purge
     $> module load OpenMPI
     $> tar xvzf /tmp/GotoBLAS2-1.13.tar.gz
     $> mv GotoBLAS2 GotoBLAS2-1.13
     […]
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


### HPL with GCC and ATLAS and MVAPICH2

Here we will rely on the [Automatically Tuned Linear Algebra Software (ATLAS)](http://math-atlas.sourceforge.net/)

Download the [latest version](http://sourceforge.net/projects/math-atlas/files/)
(3.11.17 at the time of writing) and compile them:  

     $> cd ~/TP
     $> tar xvf atlas3.11.17.tar.bz2
     $> mv ATLAS ATLAS-3.11.17 && cd ATLAS-3.11.17
     $> module purge
	 $> module load MVAPICH2	
	 $> less INSTALL.txt
	 $> mkdir build.gcc_mvapich2 && cd build.gcc_mvapich2
	 $> ../configure
	 $> grep ! ../INSTALL.txt
            make              ! tune and compile library
            make check        ! perform sanity tests
            make ptcheck      ! checks of threaded code for multiprocessor systems
            make time         ! provide performance summary as % of clock rate
            make install      ! Copy library and include files to other directories
	 $> make

Take a coffee there, it will compile for a Loooooooooooooong time

Now you can restart HPL compilation by creating (and adapting) a `Make.atlas`
and running the compilation by:  

	$> make arch=atlas

Once compiled, ensure you are able to run it: 

	$> cd bin/atlas
	$> cat HPL.dat
	$> mpirun -launcher ssh -launcher-exec /usr/bin/oarsh -hostfile $OAR_NODEFILE ./xhpl


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

