[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/parallel/basics/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/parallel/basics/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/parallel/basics/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)


# Parallel computations with OpenMP/MPI

     Copyright (c) 2013-2018 UL HPC Team  <hpc-sysadmins@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/parallel/basics/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/parallel/basics/slides.pdf)

When granted access to the [UL HPC](https://hpc.uni.lu) platform you will have at your disposal **parallel computing** resources.

Thus you will be able to run:

* ideally **parallel** (OpenMP, MPI, CUDA, OpenCL...) jobs
* however if your workflow involves **serial** tasks/jobs, you must run them *efficiently*

The objective of this tutorial is to show you how to run your MPI and/or OpenMP applications on top of the [UL HPC](https://hpc.uni.lu) platform.

For all the executions we are going to perform in this tutorial, you probably want to monitor the parallel execution on one of the allocated nodes. To do that, and **assuming** you have reserved computing resources (see the `srun / oarsub` commands of each section):

* open **another** terminal (or another `screen` window) as you'll want to monitor the execution.
* Connect to the allocated node:

```bash
############### iris cluster (slurm)
(access-iris)$> sq     # Check the allocated node
(access-iris)$> ssh iris-XXX       # ADAPT accordingly

############## gaia/chaos clusters (OAR)
(access-{gaia|chaos})$> oarstat -u     # Collect the job ID
(access-{gaia|chaos})$> oarsub -C <jobid>
```

* **For this new terminal/window**
    - run `htop`
        * press 'u' to filter by process owner, select your login
        * press 'F5' to enable the tree view

--------------------
## Pre-requisites ##

Ensure you are able to [connect to the UL HPC clusters](https://hpc.uni.lu/users/docs/access.html)
**For all tests and compilation, you MUST work on a computing node**

You'll need to prepare the data sources required by this tutorial once connected

``` bash
### ONLY if not yet done: setup the tutorials repo
# See http://ulhpc-tutorials.rtfd.io/en/latest/setup/install/
$> mkdir -p ~/git/github.com/ULHPC
$> cd ~/git/github.com/ULHPC
$> git clone https://github.com/ULHPC/tutorials.git
$> cd tutorials
$> make setup          # Initiate git submodules etc...
```

Now you can prepare a dedicated directory to work on this tutorial:

```bash
$> mkdir -p ~/tutorials/OpenMP-MPI/bin
$> cd ~/tutorials/OpenMP-MPI
$> ln -s ~/git/github.com/ULHPC/tutorials/parallel ref.d  # Symlink to the reference tutorial
$> ln -s ref.d/basics .   # Basics instructions
$> cd basics
```


--------------------------
## Parallel OpenMP Jobs ##

[OpenMP](https://www.openmp.org/) (Open Multi-Processing) is a popular parallel programming model for multi-threaded applications. More precisely, it is an Application Programming Interface (API) that supports **multi-platform shared memory multiprocessing** programming in C, C++, and Fortran on most platforms, instruction set architectures and operating systems.

* [Reference website](https://www.openmp.org/): <https://www.openmp.org/>
* __Latest version: 4.5__ (Nov 2015) -- [specifications](https://www.openmp.org/wp-content/uploads/openmp-4.5.pdf)
* Below notes are adapted from [LLNL OpenMP tutorial](https://computing.llnl.gov/tutorials/openMP/)

[OpenMP](https://www.openmp.org/) is designed for multi-processor/core, shared memory machine (nowadays NUMA). OpenMP programs accomplish parallelism **exclusively** through the use of threads.

* A __thread__ of execution is the smallest unit of processing that can be scheduled by an operating system.
    - Threads exist within the resources of a _single_ process. Without the process, they cease to exist.
* Typically, the number of threads match the number of machine processors/cores.
    - _Reminder_: **[iris](https://hpc.uni.lu/systems/iris/#computing-capacity)**: 2x14 cores; **[gaia](https://hpc.uni.lu/systems/gaia/#computing-capacity)**: depends, but typically 2x6 cores
    - However, the actual _use_ of threads is up to the application.
    - `OMP_NUM_THREADS` (if present) specifies _initially_ the _max_ number of threads;
        * you can use `omp_set_num_threads()` to override the value of `OMP_NUM_THREADS`;
        * the presence of the `num_threads` clause overrides both other values.
* OpenMP is an explicit (not automatic) programming model, offering the programmer full control over parallelization.
    - parallelization can be as simple as taking a serial program and inserting compiler directives....
    - in general, this is way more complex

* OpenMP uses the fork-join model of parallel execution
    - **FORK**: the master thread then creates a team of parallel threads.
        * The statements in the program that are enclosed by the parallel region construct are then executed in parallel among the various team threads.
    - **JOIN**: When the team threads complete the statements in the parallel region construct, they synchronize and terminate, leaving only the master thread.

![](https://upload.wikimedia.org/wikipedia/commons/f/f1/Fork_join.svg)

### Slurm reservations for OpenMP programs

* (eventually as this is the default) set a _single_ task per node with `--ntasks-per-node=1`
* Use `-c <N>` (or `--cpus-per-task <N>`) to set the number of OpenMP threads you wish to use.
* (again) **The number of threads should not exceed the number of cores on a compute node.**

Thus a minimal Slurm launcher would typically look like that -- see also [our default Slurm launchers](https://github.com/ULHPC/launcher-scripts/blob/devel/slurm/launcher.default.sh).

```bash
#!/bin/bash -l
#SBATCH --ntasks-per-node=1 # Run a single task per node, more explicit than '-n 1'
#SBATCH -c 28               #  number of CPU cores i.e. OpenMP threads per task
#SBATCH --time=0-01:00:00
#SBATCH -p batch
#SBATCH --qos=qos-batch

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Use the RESIF build modules of the UL HPC platform
if [ -f  /etc/profile ]; then
   .  /etc/profile
fi
# Load the {intel | foss} toolchain and whatever module(s) you need
module purge
module load toolchain/intel    # or toolchain/foss

srun /path/to/your/openmp_program
```

### OAR reservations for OpenMP programs

You have to setup the reservation to match the required number of OpenMP threads with the number of cores **within** the same node _i.e._

      oarsub -l nodes=1/core=<N>[...]

Thus a minimal OAR launcher would typically look like that

```bash
#!/bin/bash -l
#OAR -l nodes=1/core=4,walltime=1

export OMP_NUM_THREADS=$(cat $OAR_NODEFILE| wc -l)

# Use the RESIF build modules of the UL HPC platform
if [ -f  /etc/profile ]; then
   .  /etc/profile
fi
# Load the {intel | foss} toolchain and whatever module(s) you need
module purge
module load toolchain/intel    # or toolchain/foss

path/to/your/openmp_program
```

### OpenMP Compilation

| Toolchain         | Compilation command  |
|-------------------|----------------------|
| `toolchain/intel` | `icc -qopenmp [...]` |
| `toolchain/foss`  | `gcc -fopenmp [...]` |

### Hands-on: OpenMP Helloworld and matrix multiplication

You can find in `src/hello_openmp.c` the traditional OpenMP "Helloworld" example.

* Reserve an interactive job to launch 4 OpenMP threads (for 30 minutes)

```bash
############### iris cluster (slurm) ###############
(access-iris)$> srun -p interactive --ntasks-per-node=1 -c 4 -t 0:30:00 --pty bash
$> export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

############### gaia/chaos clusters (OAR) ###############
(access-{gaia|chaos})$> oarsub -I -l nodes=1/core=4,walltime=0:30:00
$> export OMP_NUM_THREADS=$(cat $OAR_NODEFILE| wc -l)
```

* Check the set variable `$OMP_NUM_THREADS`. Which value do you expect?

        $> echo $OMP_NUM_THREADS

* Check and compile the source `src/hello_openmp.c` to generate:
    - `bin/hello_openmp`       (compiled over the `foss`  toolchain)
    - `bin/intel_hello_openmp` (compiled over the `intel` toolchain)

```bash
$> cat src/hello_openmp.c
######### foss toolchain
$> module purge                # Safeguard
$> module load toolchain/foss
$> gcc -fopenmp -Wall -O2 src/hello_openmp.c -o bin/hello_openmp

######### intel toolchain
$> module purge                # Safeguard
$> module load toolchain/intel
$> icc -qopenmp -xhost -Wall -O2 src/hello_openmp.c -o bin/hello_openmp
```

* (__only__ if you have trouble to compile): `make openmp`

* Execute the generated binaries multiple times. What do you notice?
* Exit your interactive session (`exit` or `CTRL-D`)
* Prepare a launcher script (use your favorite editor) to execute this application in batch mode.

```bash
############### iris cluster (slurm) ###############
$> sbatch ./launcher.OpenMP.sh

############### gaia/chaos clusters (OAR) ###############
$> oarsub -S ./launcher.OpenMP.sh
```

Repeat the above procedure on a more serious computation: a naive matrix multiplication  using OpenMP, those source code is located in `src/matrix_mult_openmp.c`

Adapt the launcher script to sustain both executions (OpenMP helloworld and matrix multiplication)

_Note_: if you are lazy (or late), you can use the provided launcher script `runs/launcher.OpenMP.sh`.

```bash
$> cd runs
$> ./launcher.OpenMP.sh -h
NAME
  ./launcher.OpenMP.sh -- OpenMP launcher example
USAGE
  ./launcher.OpenMP.sh {intel | foss } [app]

Example:
  ./launcher.OpenMP.sh                          run foss on hello_openmp
  ./launcher.OpenMP.sh intel                    run intel on hello_openmp
  ./launcher.OpenMP.sh foss matrix_mult_openmp  run foss  on matrix_mult_openmp

$> ./launcher.OpenMP.sh
$> ./launcher.OpenMP.sh intel
$> ./launcher.OpenMP.sh foss matrix_mult_openmp
```

Passive jobs examples:

```bash
############### iris cluster (slurm) ###############
$> sbatch ./launcher.OpenMP.sh foss  matrix_mult_openmp
$> sbatch ./launcher.OpenMP.sh intel matrix_mult_openmp

############### gaia/chaos clusters (OAR) ###############
# Arguments of the launcher script to tests
$> cat > openmp-args.txt <<EOF
foss
intel
foss matrix_mult_openmp
intel matrix_mult_openmp
EOF

$> oarsub -S ./launcher.OpenMP.sh --array-param-file openmp-args.txt
[ADMISSION RULE] Modify resource description with type and ibpool constraints
Simple array job submission is used
OAR_JOB_ID=4357646
OAR_JOB_ID=4357647
OAR_JOB_ID=4357648
OAR_JOB_ID=4357649
OAR_ARRAY_ID=4357646
```

Check the elapsed time: what do you notice ?

### (optional) Hands-on: OpenMP data race benchmark suite

One way to test most of OpenMP feature is to evaluate its execution against a benchmark.
For instance, we are going to test OpenMP installation against  [DataRaceBench](https://github.com/LLNL/dataracebench), a benchmark suite designed to systematically and quantitatively evaluate the effectiveness of data race detection tools.
It includes a set of microbenchmarks with and without data races. Parallelism is represented by OpenMP directives.

```bash
$> cd ~/git/github.com/ULHPC/tutorials/parallel/basics
$> make fetch      # clone src/dataracebench
$> cd src/dataracebench
```

Now you can reserve the nodes and set `OMP_NUM_THREADS`:

* Reserve an interactive job to launch 12 OpenMP threads (for 30 minutes)

```bash
############### iris cluster (slurm) ###############
(access-iris)$> srun -p interactive --ntasks-per-node=1 -c 12 -t 0:30:00 --pty bash
$> export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

############### gaia/chaos clusters (OAR) ###############
(access-{gaia|chaos})$> oarsub -I -l nodes=1/core=12,walltime=0:30:00
$> export OMP_NUM_THREADS=$(cat $OAR_NODEFILE| wc -l)
```

* Open **another** terminal (or another `screen` window) to monitor the execution (see intructions on top).

* Execute the benchmark, for instance using the intel toolchain:

```bash
$> module load toolchain/intel
$> ./check-data-races.sh --help

Usage: ./check-data-races.sh [--run] [--help]

--help     : this option
--small    : compile and test all benchmarks using small parameters with Helgrind, ThreadSanitizer, Archer, Intel inspector.
--run      : compile and run all benchmarks with gcc (no evaluation)
--run-intel: compile and run all benchmarks with Intel compilers (no evaluation)
--helgrind : compile and test all benchmarks with Helgrind
--tsan     : compile and test all benchmarks with clang ThreadSanitizer
--archer   : compile and test all benchmarks with Archer
--inspector: compile and test all benchmarks with Intel Inspector

$> ./check-data-races.sh --run-intel
```

__Useful OpenMP links__:

* <https://www.openmp.org/>
* [OpenMP Tutorial LLNL](https://computing.llnl.gov/tutorials/openMP/)
* [Data race benchmark suite](https://github.com/LLNL/dataracebench)

-----------------------------------
## Parallel/Distributed MPI Jobs ##


The _Message Passing Interface_ (MPI) Standard  is a message passing library standard based on the consensus of the MPI Forum.
The goal of the Message Passing Interface is to establish a **portable**, **efficient**, and **flexible** standard for message passing that will be widely used for writing message passing programs.
MPI is not an IEEE or ISO standard, but has in fact, become the "industry standard" for writing message passing programs on HPC platforms.

* [Reference website](https://www.mpi-forum.org/): <https://www.mpi-forum.org/>
* __Latest version: 3.1__ (June 2015) -- [specifications](https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report.pdf)
* Below notes are adapted from [LLNL MPI tutorial](https://computing.llnl.gov/tutorials/mpi/)

In the MPI programming model, a computation comprises one or more **processes** that communicate by calling library routines to send and receive messages to other processes.
In most MPI implementations,   a fixed set of processes is created at program initialization, and one process is created per processor.

### MPI implementations

The [UL HPC platform](http://hpc.uni.lu) offers to you different MPI implementations:

| MPI Suit                                                        | Version | `module load`...  | Compiler                    |
|-----------------------------------------------------------------|---------|-------------------|-----------------------------|
| [Intel MPI](http://software.intel.com/en-us/intel-mpi-library/) |  17.0.1 | `toolchain/intel` | C: `mpiicc`; C++: `mpiicpc` |
| [OpenMPI](http://www.open-mpi.org/)                             |   2.1.1 | `mpi/OpenMPI`     | C: `mpicc`;  C++: `mpic++`  |
| [MVAPICH2](http://mvapich.cse.ohio-state.edu/overview/)         |    2.3a | `mpi/MVAPICH2`    | C: `mpicc`;  C++: `mpic++`  |

### MPI compilation

| MPI Suit                                                        | `module load`...  | Compilation command (C )                     |
|-----------------------------------------------------------------|-------------------|----------------------------------------------|
| [Intel MPI](http://software.intel.com/en-us/intel-mpi-library/) | `toolchain/intel` | `mpiicc -Wall [-qopenmp] [-xhost] -O2 [...]` |
| [OpenMPI](http://www.open-mpi.org/)                             | `mpi/OpenMPI`     | `mpicc  -Wall [-fopenmp] -O2 [...]`          |
| [MVAPICH2](http://mvapich.cse.ohio-state.edu/overview/)         | `mpi/MVAPICH2`    | `mpicc  -Wall [-fopenmp] -O2 [...]`          |


Of course, it is possible to have **hybrid** code, mixing MPI and OpenMP primitives.

### Slurm reservations and usage for MPI programs

* set the number of distributed nodes you want to reserver with `-N <N>`
* set the number of **MPI processes** per node (that's more explicit) with `--ntasks-per-node=<N>`
     - you can also use `-n <N>` to specify the _total_ number of MPI processes you want, but the above approach is advised.
* (eventually as this is the default) set a _single_ thread per MPI process  with `-c 1`
     - _except_ when running an hybrid code...

**Important**:

* To run your MPI program, be aware that Slurm is able to directly launch MPI tasks and initialize of MPI communications via [Process Management Interface (PMI)](https://slurm.schedmd.com/mpi_guide.html)
     - pmi2 is currently available, we'll switch to PMIx with the incoming RESIF release introducing the 2018a toolchains.
     - permits to resolve the task affinity by the scheduler (avoiding to use `mpirun --map-by [...]`)
* Simply use (whatever MPI flavor you use):

            srun -n $SLURM_NTASKS /path/to/mpiprog [...]

Thus a minimal launcher would _typically_ look like that -- see also [our default Slurm launchers](https://github.com/ULHPC/launcher-scripts/blob/devel/slurm/launcher.default.sh).

```bash
#!/bin/bash -l
#SBATCH -N 2                  # Use 2 nodes
#SBATCH --ntasks-per-node=28  # Number of MPI process per node
#SBATCH -c 1        # Number of threads per MPI process (1 unless hybrid code)
#SBATCH --time=0-01:00:00
#SBATCH -p batch
#SBATCH --qos=qos-batch

# Use the RESIF build modules of the UL HPC platform
if [ -f  /etc/profile ]; then
   .  /etc/profile
fi
# Load the intel toolchain and whatever MPI module you need
module purge
module load toolchain/intel    # or mpi/{OpenMPI|MVAPICH2}
# export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
srun -n $SLURM_NTASKS /path/to/your/mpi_program
```

In the above example, 2x28 = 56 MPI processes will be launched.

### OAR reservations and usage for MPI programs

You have to setup the reservation to match the required number of MPI processes:

      oarsub -l nodes=2/core=12[...]

As for running you MPI program, you typically need to rely on `mpirun` for which the command-line (unfortunately) differs depending on the MPI flavor:

| MPI Suit                                                        | `module load`...  | Typical run command (OAR)                                                   |
|-----------------------------------------------------------------|-------------------|-----------------------------------------------------------------------------|
| [Intel MPI](http://software.intel.com/en-us/intel-mpi-library/) | `toolchain/intel` | `mpirun -hostfile $OAR_NODEFILE [...]`                                      |
| [OpenMPI](http://www.open-mpi.org/)                             | `mpi/OpenMPI`     | `mpirun -hostfile $OAR_NODEFILE -x PATH -x LD_LIBRARY_PATH [...]`           |
| [MVAPICH2](http://mvapich.cse.ohio-state.edu/overview/)         | `mpi/MVAPICH2`    | `mpirun -launcher ssh -launcher-exec /usr/bin/oarsh -f $OAR_NODEFILE [...]` |


Thus a minimal OAR launcher would typically look like that

```bash
#!/bin/bash -l
#OAR -l nodes=2/core=6,walltime=1

# Use the RESIF build modules of the UL HPC platform
if [ -f  /etc/profile ]; then
   .  /etc/profile
fi
# Load the intel toolchain and whatever MPI module you need
module purge
module load toolchain/intel    # or mpi/{OpenMPI|MVAPICH2}
# ONLY on moonshot node that have no IB card: export I_MPI_FABRICS=tcp

### Intel MPI
mpirun -hostfile $OAR_NODEFILE path/to/mpi_program
### OpenMPI
mpirun -hostfile $OAR_NODEFILE -x PATH -x LD_LIBRARY_PATH path/to/mpi_program
### MVAPICH2
mpirun -launcher ssh -launcher-exec /usr/bin/oarsh -f $OAR_NODEFILE path/to/mpi_program
```

### Hands-on: MPI Helloworld and matrix multiplication

You can find in `src/hello_mpi.c` the traditional MPI "Helloworld" example.

* Reserve an interactive job to launch 6 MPI processes across two nodes 2x3 (for 30 minutes)

```bash
############### iris cluster (slurm) ###############
(access-iris)$> srun -p interactive -N 2 --ntasks-per-node=3 -t 0:30:00 --pty bash

############### gaia/chaos clusters (OAR) ###############
(access-{gaia|chaos})$> oarsub -I -l nodes=2/core=3,walltime=0:30:00
```

* Check and compile the source `src/hello_mpi.c` to generate:
    - `bin/openmpi_hello_mpi`  (compiled with the `mpi/OpenMPI` module)
    - `bin/intel_hello_mpi`    (compiled over the `intel` toolchain and Intel MPI)
    - `bin/mvapich2_hello_mpi` (compiled over the `mpi/MVAPICH2` toolchain)

```bash
$> cat src/hello_mpi.c
######### OpenMPI
$> module purge                # Safeguard
$> module load mpi/OpenMPI
$> mpicc -Wall -O2 src/hello_mpi.c -o bin/openmpi_hello_mpi

######### Intel MPI
$> module purge                # Safeguard
$> module load toolchain/intel
$> mpiicc -Wall -xhost -O2 src/hello_mpi.c -o bin/intel_hello_mpi

######### MVAPICH2
$> module purge                # Safeguard
$> module load mpi/MVAPICH2
$> mpicc -Wall -O2 src/hello_mpi.c -o bin/mvapich2_hello_mpi
```

* (__only__ if you have trouble to compile): `make mpi`
* Execute the generated binaries multiple times. What do you notice?
* Exit your interactive session (`exit` or `CTRL-D`)
* Prepare a launcher script (use your favorite editor) to execute this application in batch mode.

```bash
############### iris cluster (slurm) ###############
$> sbatch ./launcher.MPI.sh

############### gaia/chaos clusters (OAR) ###############
$> oarsub -S ./launcher.MPI.sh
```

Repeat the above procedure on a more serious computation: a naive matrix multiplication  using MPI, those source code is located in `src/matrix_mult_mpi.c`

Adapt the launcher script to sustain both executions (MPI helloworld and matrix multiplication)

_Note_: if you are lazy (or late), you can use the provided launcher script `runs/launcher.MPI.sh`.

```bash
$> cd runs
$> ./launcher.MPI.sh -h
NAME
  ./launcher.MPI.sh -- MPI launcher example
USAGE
  ./launcher.MPI.sh {intel | openmpi | mvapich2} [app]

Example:
  ./launcher.MPI.sh                          run OpenMPI on hello_mpi
  ./launcher.MPI.sh intel                    run Intel MPI on hello_mpi
  ./launcher.MPI.sh openmpi matrix_mult_mpi  run OpenMPI on matrix_mult_mpi
```

Passive jobs examples:

```bash
############### iris cluster (slurm) ###############
$> sbatch ./launcher.MPI.sh openmpi matrix_mult_mpi
$> sbatch ./launcher.MPI.sh intel   matrix_mult_mpi

############### Gaia/chaos clusters (OAR) ###############
# Arguments of the launcher script to tests
$> cat > mpi-args.txt <<EOF
openmpi matrix_mult_mpi
intel   matrix_mult_mpi
EOF

$> oarsub -S ./launcher.MPI.sh --array-param-file mpi-args.txt
[ADMISSION RULE] Modify resource description with type and ibpool constraints
Simple array job submission is used
OAR_JOB_ID=4357652
OAR_JOB_ID=4357653
OAR_ARRAY_ID=4357652
```

Check the elapsed time: what do you notice ?

__Useful MPI links__:

* <http://www.mpi-forum.org/docs/>
* [MPI Tutorial LLNL](https://computing.llnl.gov/tutorials/mpi/)
* [Intel MPI](http://software.intel.com/en-us/intel-mpi-library/):
   - [Step by Step Performance Optimization with Intel® C++ Compiler](https://software.intel.com/en-us/articles/step-by-step-optimizing-with-intel-c-compiler)
   - [Intel(c) C++ Compiler 17.0 Developer Guide and Reference](https://software.intel.com/en-us/intel-cplusplus-compiler-17.0-user-and-reference-guide) (`toolchain/intel/2017a`)

--------------------------------
## Hybrid OpenMP+MPI Programs ##

Of course, you can have _hybrid_ code mixing MPI and OpenMP primitives.

* You need to compile the code with the `-qopenmp` (with Intel MPI) or `-fopenmp` (for the other MPI suits) flags
* You need to adapt the `OMP_NUM_THREADS` environment variable accordingly
* __(Slurm only)__: you need to adapt the value `-c <N>` (or `--cpus-per-task <N>`) to set the number of OpenMP threads you wish to use per MPI process
*  **(OAR only)**: you have to take the following elements into account:
    - You need to compute accurately the number of MPI processes per node `<PPN>` (in addition to the number of MPI processes) and pass it to `mpirun`
        * OpenMPI:   `mpirun -npernode <PPN> -np <N>`
        * Intel MPI: `mpirun -perhost <PPN>  -np <N>`
        * MVAPICH2:  `mpirun -ppn <PPN>      -np <N>`
    - You need to ensure the environment variable `OMP_NUM_THREADS` is shared across the nodes
    - (_Intel MPI only_) you probably want to set [`I_MPI_PIN_DOMAIN=omp`](https://software.intel.com/en-us/mpi-developer-reference-linux-interoperability-with-openmp-api)
    - (_MVAPICH2 only_) you probably want to set `MV2_ENABLE_AFFINITY=0`

### Slurm launcher for OpenMP+MPI programs

```bash
#!/bin/bash -l
#SBATCH -N 2                  # Use 2 nodes
#SBATCH --ntasks-per-node=1   # Number of MPI process per node
#SBATCH -c 4                  # Number of OpenMP threads per MPI process
#SBATCH --time=0-01:00:00
#SBATCH -p batch
#SBATCH --qos=qos-batch

# Use the RESIF build modules of the UL HPC platform
if [ -f  /etc/profile ]; then
   .  /etc/profile
fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Load the intel toolchain and whatever MPI module you need
module purge
module load toolchain/intel    # or mpi/{OpenMPI|MVAPICH2}
# export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
srun -n $SLURM_NTASKS /path/to/your/hybrid_program
```

### OAR launcher for OpenMP+MPI programs

```bash
#!/bin/bash -l
#OAR -l nodes=2/core=4,walltime=1

# Use the RESIF build modules of the UL HPC platform
if [ -f  /etc/profile ]; then
   .  /etc/profile
fi

export OMP_NUM_THREADS=$(cat $OAR_NODEFILE | uniq -c | head -n1 | awk '{print $1}')

NTASKS=$(cat $OAR_NODEFILE | wc -l)
NNODES=$(cat $OAR_NODEFILE | sort -u | wc -l)
NCORES_PER_NODE=$(echo "${NTASKS}/${NNODES}" | bc)
NPERNODE=$(echo "$NCORES_PER_NODE/$OMP_NUM_THREADS" | bc)
NP=$(echo "$NTASKS/$OMP_NUM_THREADS" | bc)

# Unique list of hostname for the machine file
MACHINEFILE=hostfile_${OAR_JOBID}.txt;
cat $OAR_NODEFILE | uniq > ${MACHINEFILE};

### Load the intel toolchain and whatever MPI module you need
module purge
module load toolchain/intel    # or mpi/{OpenMPI|MVAPICH2}
# ONLY on moonshot node that have no IB card: export I_MPI_FABRICS=tcp

### Intel MPI
mpirun -perhost ${NPERNODE:=1} -np ${NP} \
       -genv OMP_NUM_THREADS=${OMP_NUM_THREADS} -genv I_MPI_PIN_DOMAIN=omp \
       -hostfile $OAR_NODEFILE  path/to/hybrid_program

### OpenMPI
mpirun -npernode ${NPERNODE:=1} -np ${NP} \
       -x OMP_NUM_THREADS -x PATH-x LD_LIBRARY_PATH \
       -hostfile $OAR_NODEFILE  path/to/hybrid_program

### MVAPICH2
export MV2_ENABLE_AFFINITY=0
mpirun -ppn ${NPERNODE:=1} -np ${NP} -genv OMP_NUM_THREADS=${OMP_NUM_THREADS} \
       -launcher ssh -launcher-exec /usr/bin/oarsh \
       -f $MACHINEFILE  path/to/hybrid_program
```

### Hands-on: Hybrid OpenMP+MPI Helloworld

You can find in `src/hello_hybrid.c` the traditional OpenMP+MPI "Helloworld" example.

* Reserve an interactive job to launch 2 MPI processes (1 per node), each composed of 4 OpenMP threads (for 30 minutes)

```bash
############### iris cluster (slurm) ###############
(access-iris)$> srun -p interactive -N 2 --ntasks-per-node=1 -c 4 -t 0:30:00 --pty bash
$> export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

############### gaia/chaos clusters (OAR) ###############
(access-{gaia|chaos})$> oarsub -I -l nodes=2/core=4,walltime=0:30:00
$> export OMP_NUM_THREADS=$(cat $OAR_NODEFILE | uniq -c | head -n1 | awk '{print $1}')
```

* Check the set variable `$OMP_NUM_THREADS`. Which value do you expect?

        $> echo $OMP_NUM_THREADS

* Check and compile the source `src/hello_hybrid.c` to generate:
    - `bin/openmpi_hello_hybrid`  (compiled with the `mpi/OpenMPI` module)
    - `bin/intel_hello_hybrid`    (compiled over the `intel` toolchain and Intel MPI)
    - `bin/mvapich2_hello_hybrid` (compiled over the `mpi/MVAPICH2` toolchain)

```bash
$> cat src/hello_hybrid.c
######### OpenMPI
$> module purge                # Safeguard
$> module load mpi/OpenMPI
$> mpicc -fopenmp -Wall -O2 src/hello_hybrid.c -o bin/openmpi_hello_hybrid

######### Intel MPI
$> module purge                # Safeguard
$> module load toolchain/intel
$> mpiicc -qopenmp -Wall -xhost -O2 src/hello_hybrid.c -o bin/intel_hello_hybrid

######### MVAPICH2
$> module purge                # Safeguard
$> module load mpi/MVAPICH2
$> mpicc -f openmp -Wall -O2 src/hello_hybrid.c -o bin/mvapich2_hello_hybrid
```

* (__only__ if you have trouble to compile): `make hybrid`
* Execute the generated binaries (see above tips)
* Exit your interactive session (`exit` or `CTRL-D`)
* Adapt the MPI launcher to allow for batch jobs submissions over hybrid programs

```bash
############### iris cluster (slurm) ###############
$> sbatch ./launcher.hybrid.sh

############### gaia/chaos clusters (OAR) ###############
$> oarsub -S ./launcher.hybrid.sh
```

_Note_: if you are lazy (or late), you can use the provided launcher script `runs/launcher.MPI.sh`.

```bash
$> cd runs
$> ./launcher.hybrid.sh -h
NAME
  ./launcher.hybrid.sh -- Hybrid OpenMP+MPI launcher example
USAGE
  ./launcher.hybrid.sh {intel | openmpi | mvapich2} [app]

Example:
  ./launcher.hybrid.sh [openmpi]      run hybrid OpenMP+OpenMPI  on hello_hybrid
  ./launcher.hybrid.sh intel          run hybrid OpenMP+IntelMPI on hello_hybrid
  ./launcher.hybrid.sh mvapich2       run hybrid OpenMP+MVAPICH2 on hello_hybrid
```

Passive jobs examples:

```bash
############### iris cluster (slurm) ###############
$> sbatch ./launcher.hybrid.sh
$> sbatch ./launcher.hybrid.sh intel
$> sbatch ./launcher.hybrid.sh mvapich2

############### Gaia/chaos clusters (OAR) ###############
# Arguments of the launcher script to tests
$> cat > hybrid-args.txt <<EOF
openmpi
intel
EOF
$> oarsub -S ./launcher.hybrid.sh --array-param-file hybrid-args.txt
[ADMISSION RULE] Modify resource description with type and ibpool constraints
Simple array job submission is used
OAR_JOB_ID=4357843
OAR_JOB_ID=4357844
OAR_ARRAY_ID=4357843
```

----------------------------------------------------------------
## Code optimization tips for your OpenMP and/or MPI programs ##

* Consider changing your memory allocation functions to avoid fragmentation and enable scalable concurrency support (this applies for OpenMP and/or MPI programs)
     - Facebook's [jemalloc](http://jemalloc.net/)
     - Google's [tcmalloc](https://github.com/gperftools/gperftools)

* When using the `intel` toolchain:
     -  see the [Step by Step Performance Optimization with Intel(c) C++ Compiler](https://software.intel.com/en-us/articles/step-by-step-optimizing-with-intel-c-compiler)
         * the `-xhost` option permits to enable  processor-specific optimization.
         * you might wish to consider Interprocedural Optimization (IPO) approach, an automatic, multi-step process that allows the compiler to analyze your code to determine where you can benefit from specific optimizations.

---------------------
## Troubleshooting ##

* `srun: error: PMK_KVS_Barrier duplicate request from task ...`
   - you are trying to use `mpirun` (instead of `srun`) from Intel MPI within a SLURM session and receive such error on `mpirun`:  make sure `$I_MPI_PMI_LIBRARY` is **not** set (`unset I_MPI_PMI_LIBRARY``).
