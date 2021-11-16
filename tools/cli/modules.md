[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/beginners/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/beginners/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/beginners/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Using software modules

     Copyright (c) 2013-2020 UL HPC Team <hpc-sysadmins@uni.lu>

This page is part of the Getting started tutorial.

[Environment Modules](http://modules.sourceforge.net/) is a software package that allows us to provide a [multitude of applications and libraries in multiple versions](http://hpc.uni.lu/users/software/) on the UL HPC platform. The tool itself is used to manage environment variables such as `PATH`, `LD_LIBRARY_PATH` and `MANPATH`, enabling the easy loading and unloading of application/library profiles and their dependencies.

We will have multiple occasion to use modules in the other tutorials so there is nothing special we foresee here. You are just encouraged to read the following resources:

* [Introduction to Environment Modules by Wolfgang Baumann](https://www.hlrn.de/home/view/System3/ModulesUsage)
* [Modules tutorial @ NERSC](https://www.nersc.gov/users/software/nersc-user-environment/modules/)
* [UL HPC documentation on modules](https://hpc.uni.lu/users/docs/modules.html)


## Commands

By loading appropriate environment modules, the user can select:

* compilers,
* libraries, e.g. the MPI library, or
* other third party software packages.

An exhaustive list of the available software is proposed [in this page](https://hpc.uni.lu/users/software/).

On a node, using an interactive jobs, you can:

* list all available softwares: `module avail`
* search for one software: `module spider <search terms>`
* "load" a software in your environment: `module load <category>/<software>[/<version>]`
* list the currently loaded modules: `module list`
* clean your environment, unload everything: `module purge`


## Software sets

Currently, the ULHPC provides the software sets 2019a (default) and 2019b (devel).
We encourage you to use right now 2019b, by redefining the `MODULEPATH` variable as explained in the next section,
as it will soon be promoted as the next default environment.

The ULHPC team updates the software set every year based on the Easybuild releases.

| Name    | Type      | 2019b (legacy)     | 2020a        | 2020b (prod) | 2021a (devel) |
|---------|-----------|--------------------|--------------|--------------|---------------|
| GCCCore | compiler  | 8.3.0              | 9.3.0        | 10.2.0       | 10.3.0        |
| foss    | toolchain | 2019b              | 2020a        | 2020b        | 2021a         |
| intel   | toolchain | 2019b              | 2020a        | 2020b        | 2021a         |
| Python  |           | 3.7.4 (and 2.7.16) | 3.8.2        | 3.8.6        | 3.9.2         |

Each environment provides different versions of softwares.
The core of the software environment is the toolchain, a toolchain is a set of tools used to compile and run of the other programs of the software environment.

The ULHPC team provides two toolchain:

* `foss` : based on open source software (`GCC`, `binutils`, `OpenMPI`, `OpenBLAS`, `FFTW`)
* `intel`: based on the proprietary Intel compiler suite.

## `$MODULEPATH` environment variable

The environment variable `$MODULEPATH` contains the path of the directory containing the modules.
You can use this variable to change your software environment.


In the following command, we use the 2020b environment compiled for Broadwell CPUs on Iris.

```
export MODULEPATH=/opt/apps/resif/iris/2020b/broadwell/modules/all
```

This is equivalent to the command `resif-load-swset-devel` on Iris.

For backward compatibility reasons and for reproducibility, it is always possible to load the older environments, with the command `resif-load-swset-legacy`

In order to restore production settings, run the command `resif-load-swset-prod`

## Examples

Choose one or two languages below, and try to run the hello world program on a compute node.


### Matlab

1. Create a file named `fibonacci.m` in your home directory, copy-paste the following code in this file.
   This code will calculate the first N numbers of the Fibonacci sequence


        N=1000;
        fib=zeros(1,N);
        fib(1)=1;
        fib(2)=1;
        k=3;
        while k <= N
          fib(k)=fib(k-2)+fib(k-1);
          fprintf('%d\n',fib(k));
          pause(1);
          k=k+1;
        end


2. Create a new interactive job

3. Look for the `matlab` module using the command `module spider`

4. Load the module `base/MATLAB` using the command `module load`

5. Execute the code using matlab

        (node)$> matlab -nojvm -nodisplay -nosplash < path/to/fibonacci.m


### R

1. Create a file named `fibonacci.R` in your home directory, copy-paste the following code in this file.
   This code will calculate the first N numbers of the Fibonacci sequence


        N <- 130
        fibvals <- numeric(N)
        fibvals[1] <- 1
        fibvals[2] <- 1
        for (i in 3:N) {
             fibvals[i] <- fibvals[i-1]+fibvals[i-2]
             print( fibvals[i], digits=22)
             Sys.sleep(1)
        }

2. Create a new interactive job

3. Look for the `R` module using the command `module spider`

3. Load the module `lang/R` using the command `module load`

4. Execute the code using R

        (node)$> Rscript path/to/fibonacci.R



### C

Create a new file called `helloworld.c`, containing the source code of a simple "Hello World" program written in C.


        #include<stdio.h>

        int main()
        {
            printf("Hello, world!");
            return 0;
        }


First, compile the program using the "FOSS" toolchain, containing the GNU C compiler

        (node)$> module load toolchain/foss
        (node)$> gcc helloworld.c -o helloworld

Then, compile the program using the Intel toolchain, containing the ICC compiler

        (node)$> module purge
        (node)$> module load toolchain/intel
        (node)$> icc helloworld.c -o helloworld

If you use Intel CPUs and ICC is available on the platform, it is advised to use ICC in order to produce optimized binaries and achieve better performance.


### C++

**Question:** create a new file `helloworld.cpp` containing the following C++ source code,
compile the following program, using GNU C++ compiler (`g++` command), and the Intel compiler (`icpc` command).


        #include <iostream>

        int main() {
            std::cout << "Hello, world!" << std::endl;
        }



### Fortran

**Question:** create a new file `helloworld.f` containing the following source code,
compile the following program, using the GNU Fortran compiler (`gfortran` command), and ICC (`ifortran` command).


        program hello
           print *, "Hello, World!"
        end program hello


Be careful, the 6 spaces at the beginning of each line are required



### MPI

MPI is a programming interface that enables the communication between processes of a distributed memory system.

We will create a simple MPI program where the MPI process of rank 0 broadcasts an integer (42) to all the other processes.
Then, each process prints its rank, the total number of processes and the value he received from the process 0.

In your home directory, create a file `mpi_broadcast.c` and copy the following source code:


        #include <stdio.h>
        #include <mpi.h>
        #include <unistd.h>
        #include <time.h> /* for the work function only */

        int main (int argc, char *argv []) {
               char hostname[257];
               int size, rank;
               int i, pid;
               int bcast_value = 1;

               gethostname(hostname, sizeof hostname);
               MPI_Init(&argc, &argv);
               MPI_Comm_rank(MPI_COMM_WORLD, &rank);
               MPI_Comm_size(MPI_COMM_WORLD, &size);
               if (!rank) {
                    bcast_value = 42;
               }
               MPI_Bcast(&bcast_value,1 ,MPI_INT, 0, MPI_COMM_WORLD );
               printf("%s\t- %d - %d - %d\n", hostname, rank, size, bcast_value);
               fflush(stdout);

               MPI_Barrier(MPI_COMM_WORLD);
               MPI_Finalize();
               return 0;
        }

Reserve 2 tasks of 1 core on two distinct nodes with Slurm

        (access-iris)$> si --time 1:00:0 -N 2 -n 2 -c 1

Load a toolchain and compile the code using `mpicc`

        (node)$> mpicc mpi_broadcast.c -o mpi_broadcast -lpthread

With Slurm, you can use the `srun` command. Create an interactive job, with 2 nodes (`-N 2`), and at least 2 tasks (`-n 2`).

        (node)$> srun -n $SLURM_NTASKS ~/mpi_broadcast

