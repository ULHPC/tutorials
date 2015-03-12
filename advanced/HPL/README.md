-*- mode: markdown; mode: auto-fill; fill-column: 80 -*-

`README.md`

Copyright (c) 2013 Sebastien Varrette <Sebastien.Varrette@uni.lu>

        Time-stamp: <Dim 2013-11-10 19:21 svarrette>

-------------------


# UL HPC Tutorial: HPL benchmarking on UL HPC platform

The objective of this tutorial is to compile and run on of the reference HPC
benchmarks, [HPL](http://www.netlib.org/benchmark/hpl/), on top of the
[UL HPC](http://hpc.uni.lu) platform.  

You can work in groups for this training, yet individual work is encouraged to
ensure you understand and practice the usage of an HPC platform.  

In all cases, ensure you are able to [connect to the chaos and gaia cluster](https://hpc.uni.lu/users/docs/access.html). 

	/!\ FOR ALL YOUR COMPILING BUSINESS, ENSURE YOU WORK ON A COMPUTING NODE
	
	(access)$> 	oarsub -I -l nodes=1,walltime=4

**Advanced users only**: rely on `screen` (see
  [tutorial](http://support.suso.com/supki/Screen_tutorial)) on the frontend
  prior to running any `oarsub` command to be more resilient to disconnection.  

The latest version of this tutorial is available on
[Github](https://github.com/ULHPC/tutorials/tree/devel/advanced/HPL)

## Objectives

[HPL](http://www.netlib.org/benchmark/hpl/) is a portable implementation of
HPLinpack used to provide data for the [Top500](http://top500.org) list  

HPL rely on an efficient implementation of the Basic Linear Algebra Subprograms
(BLAS). You have several choices at this level: 

* Intel MKL
* [ATLAS](http://math-atlas.sourceforge.net/atlas_install/)
* [GotoBlas](https://www.tacc.utexas.edu/research-development/tacc-software/gotoblas2/)  

The objective of this practical session is to compare the performances of HPL
runs compiled under different combination:  

1. HPL + Intel MKL + Intel MPI
2. HPL + GCC + GotoBLAS2 + Open MPI
3. HPL + GCC + ATLAS + MPICH2

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
    $> module avail 2>&1 | grep -i MPI
    $> module load ictce
    $> module list
	Currently Loaded Modulefiles:
			1) icc/2013.5.192     2) ifort/2013.5.192   3) impi/4.1.1.036     4) imkl/11.0.5.192    5) ictce/5.5.0
	$> module show impi
	$> module show imkl

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
		LAinc        = -I$(LAdir)/mkl/include
		LAlib        = -L$(LAdir)/mkl/lib/intel64 -Wl,--start-group $(LAdir)/mkl/lib/intel64/libmkl_intel_lp64.a $(LAdir)/mkl/lib/intel64/libmkl_intel_thread.a $(LAdir)/mkl/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread

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
(3.11.32 at the time of writing) and compile it:  

     $> cd ~/TP
     $> tar xvf atlas3.11.32.tar.bz2
     $> mv ATLAS ATLAS-3.11.32 && cd ATLAS-3.11.32
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

