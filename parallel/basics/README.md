<!-- [![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/parallel/basics/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/parallel/basics/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/parallel/basics/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials) -->


<!-- # Parallel computations with OpenMP/MPI -->

<!--      Copyright (c) 2013-2018 UL HPC Team  <hpc-sysadmins@uni.lu> -->

<!-- [![](https://github.com/ULHPC/tutorials/raw/devel/parallel/basics/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/parallel/basics/slides.pdf) -->

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


-----------------------------------
## Threaded Parallel OpenMP Jobs ##

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

Thus a minimal launcher would typically look like that -- see also [our default Slurm launchers](https://github.com/ULHPC/launcher-scripts/blob/devel/slurm/launcher.default.sh).

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

Thus a minimal launcher would typically look like that

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

### Hands-on: OpenMP Helloworld

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
  $> icc -qopenmp -Wall -O2 src/hello_openmp.c -o bin/hello_openmp
  ```
* (__only__ if you have trouble to compile): `make openmp`

* Execute the generated binaries multiple times. What do you notice?
* Prepare a launcher script (use your favorite editor) to execute this application in batch mode.
  ```bash
  ############### iris cluster (slurm) ###############
  $> sbatch ./launcher.OpenMP.sh

  ############### gaia/chaos clusters (OAR) ###############
  $> oarsub -S ./launcher.OpenMP.sh
  ```




### Data race benchmark suite

One way to test most of OpenMP feature is to evaluate its execution against a benchmark.
For instance, we are going to test OpenMP installation against  [DataRaceBench](https://github.com/LLNL/dataracebench), a benchmark suite designed to systematically and quantitatively evaluate the effectiveness of data race detection tools.
It includes a set of microbenchmarks with and without data races. Parallelism is represented by OpenMP directives.

```bash
$> mkdir ~/tutorials/OpenMP
$> cd  ~/tutorials/OpenMP
$> git clone https://github.com/LLNL/dataracebench.git
$> cd dataracebench.git
```

Now you can reserve the nodes and set `OMP_NUM_THREADS`:

```bash
# /!\ FOR ALL YOUR COMPILING BUSINESS, ENSURE YOU WORK ON A COMPUTING NODE
# Have an interactive job
############### iris cluster (slurm) ###############
(access-iris)$> si -N 1 -c 28 -t 0:10:00  # 10 min reservation
# OR (long version)
(access-iris)$> srun -p interactive -N 1 -c 14 -t 0:10:00 --pty bash

$> export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

############### gaia/chaos clusters (OAR) ###############
(access-{gaia|chaos})$> oarsub -I -l nodes=1/core=12,walltime=1
$> export OMP_NUM_THREADS=$(cat $OAR_NODEFILE | wc -l)
```

Open **another** terminal (or another `screen` window) to monitor the execution (see intructions on top).

Execute the benchmark, for instance using the intel toolchain:

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
