[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/parallel/mpi/HPL/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/parallel/mpi/HPL/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# High-Performance Linpack (HPL) benchmarking on UL HPC platform

     Copyright (c) 2013-2019 UL HPC Team  <hpc-sysadmins@uni.lu>

The objective of this tutorial is to compile and run on of the reference HPC benchmarks, [HPL](http://www.netlib.org/benchmark/hpl/), on top of the [UL HPC](http://hpc.uni.lu) platform.

You can work in groups for this training, yet individual work is encouraged to ensure you understand and practice the usage of MPI programs on an HPC platform.
If not yet done, you should consider completing the following tutorials:

1. [Parallel computations with OpenMP/MPI](../../basics/) covering the basics for OpenMP, MPI or Hybrid OpenMP+MPI runs
2. [OSU Micro-benchmark](../OSU_MicroBenchmarks/)

__Resources__

* [Tweak HPL parameters](http://www.advancedclustering.com/act_kb/tune-hpl-dat-file/)
* [HPL Calculator](http://hpl-calculator.sourceforge.net/) to find good parameters
and expected performances
* [Intel Math Kernel Library Link Line Advisor](https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor)

--------------------
## Pre-requisites ##

Ensure you are able to [connect to the UL HPC clusters](https://hpc-docs.uni.lu/connect/access/).
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

If you previously cloned this repository, you probably want to collected the latest commits using:

``` bash
$> cd ~/git/github.com/ULHPC/tutorials
$> git pull     # OR (better)
$> make up
```

Now you can prepare a dedicated directory to work on this tutorial:

```bash
$> mkdir -p ~/tutorials/HPL
$> cd ~/tutorials/HPL
# Keep a symbolic link 'ref.ulhpc.d' to the reference tutorial
$> ln -s ~/git/github.com/ULHPC/tutorials/parallel/mpi/HPL ref.ulhpc.d
$> ln -s ref.ulhpc.d/Makefile .     # symlink to the root Makefile
```

**Advanced users only**: rely on `screen` (see  [tutorial](http://support.suso.com/supki/Screen_tutorial) or the [UL HPC tutorial](https://hpc.uni.lu/users/docs/ScreenSessions.html) on the  frontend prior to running any `srun/sbatch` command to be more resilient to disconnection.

Finally, be aware that the latest version of this tutorial is available on
[Github](https://github.com/ULHPC/tutorials/tree/devel/parallel/mpi/HPL/) and on

<http://ulhpc-tutorials.readthedocs.io/en/latest/parallel/mpi/HPL/>


#### Theoretical Peak Performance R<sub>peak</sub>

The ULHPC computing nodes feature the following types of processors (see also `/etc/motd` on the access node):

| Vendor | Model                          | #cores | TDP  | Freq.  | AVX512 Freq | Nodes                    | R<sub>peak</sub> |
|--------|--------------------------------|--------|------|--------|-------------|--------------------------|------------------|
| Intel  | Xeon E5-2680v4 (broadwell)     |     14 | 120W | 2.4Ghz | n/a         | `iris-[001-108]`         | 537,6  GFlops    |
| Intel  | Xeon Gold 6132 (skylake)       |     14 | 140W | 2.6GHz | 2.3 GHz     | `iris-[109-186,191-196]` | 1030,4 GFlops    |
| Intel  | Xeon Platinum 8180M  (skylake) |     28 | 205W | 2.5GHz | 2.3 GHz     | `iris-[187-190]`         | 2060,8 GFlops    |

Computing the theoretical peak performance of these processors is done using the following formulae:

R<sub>peak</sub> = #Cores x [AVX512 All cores Turbo] Frequency x #DP_ops_per_cycle

Knowing that:

* Broadwell processors (`iris-[001-108]` nodes) carry on 16 DP ops/cycle and supports AVX2/FMA3.
* Skylake   processors (`iris-[109-196]` nodes) belongs to the Gold or Platinum family and thus have two AVX512 units, thus they are capable of performing 32 Double Precision (DP) Flops/cycle. From the [reference Intel documentation](
https://www.intel.com/content/dam/www/public/us/en/documents/specification-updates/xeon-scalable-spec-update.pdf), it is possible to extract for the featured model the AVX-512 Turbo Frequency (i.e., the maximum core frequency in turbo mode) in place of the base non-AVX core frequency that can be used to compute the peak performance (see Fig. 3 p.14).

HPL permits to measure the **effective** R<sub>max</sub> performance (as opposed to the above **peak** performance R<sub>peak</sub>).
The ratio R<sub>max</sub>/R<sub>peak</sub> corresponds to the _HPL efficiency_.

----------------
## Objectives ##

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

For the sake of time and simplicity, we will focus on the combination expected to lead to the best performant runs, _i.e._ Intel MKL and Intel MPI suite, either in full MPI or in hybrid run (on 1 or 2 nodes).
As a bonus, comparison with the reference HPL binary compiled as part of the `toolchain/intel` will be considered.

------------------------------
## Fetching the HPL sources ##

In the working directory `~/tutorials/HPL`, fetch and uncompress the latest version of the [HPL](http://www.netlib.org/benchmark/hpl/) benchmark (_i.e._ **version 2.3** at the time of writing).

```bash
$> cd ~/tutorials/HPL
$> mkdir src
# Download the sources
$> cd src
# Download the latest version
$> export HPL_VERSION=2.3
$> wget --no-check-certificate http://www.netlib.org/benchmark/hpl/hpl-${HPL_VERSION}.tar.gz
$> tar xvzf hpl-${HPL_VERSION}.tar.gz
$> cd  hpl-${HPL_VERSION}
```

_Alternatively_, you can use the following command to fetch and uncompress the HPL sources:

``` bash
$> cd ~/tutorials/HPL
$> make fetch
$> make uncompress
```

--------------------------------
## Building the HPL benchmark ##

We are first going to use the [Intel Cluster Toolkit Compiler Edition](http://software.intel.com/en-us/intel-cluster-toolkit-compiler/), which provides Intel C/C++ and Fortran compilers, Intel MPI.

```bash
$> cd ~/tutorials/HPL
# Copy the provided Make.intel64
$> cp ref.ulhpc.d/src/Make.intel64 src/
```

Now you can reserve an interactive job for the compilation **from the access** server:

``` bash
# Quickly get one interactive job for 1h
$> si -N 2 --ntasks-per-node 2
# OR get one interactive (totalling 2*2 MPI processes) on broadwell-based nodes
$> salloc -p interactive -N 2 -C broadwell --ntasks-per-node 2
# OR get one interactive (totalling 2*2 MPI processes) on skylake-based nodes
$> salloc -p interactive -N 2 -C skylake --ntasks-per-node 2
```

Now that you are on a computing node, you can load the appropriate module  for Intel MKL and Intel MPI suite, i.e. `toolchain/intel`:


``` bash
# Load the appropriate module
$> module load toolchain/intel
$> module list
Currently Loaded Modules:
Currently Loaded Modules:
  1) compiler/GCCcore/6.4.0
  2) tools/binutils/2.28-GCCcore-6.4.0
  3) compiler/icc/2018.1.163-GCC-6.4.0-2.28
  4) compiler/ifort/2018.1.163-GCC-6.4.0-2.28
  5) toolchain/iccifort/2018.1.163-GCC-6.4.0-2.28
  6) mpi/impi/2018.1.163-iccifort-2018.1.163-GCC-6.4.0-2.28
  7) toolchain/iimpi/2018a
  8) numlib/imkl/2018.1.163-iimpi-2018a
  9) toolchain/intel/2018a
```

Eventually, if you want to load the development branch of the modules provided on the UL HPC platform, proceed as follows:

``` bash
$> module purge
$> module load swenv/default-env/devel
Module warning: The development software environment is not guaranteed to be stable!
$> module load toolchain/intel
$> module list
Currently Loaded Modules:
  1) swenv/default-env/devel
  2) compiler/GCCcore/8.2.0
  3) lib/zlib/1.2.11-GCCcore-8.2.0
  4) tools/binutils/2.31.1-GCCcore-8.2.0
  5) compiler/icc/2019.1.144-GCC-8.2.0-2.31.1
  6) compiler/ifort/2019.1.144-GCC-8.2.0-2.31.1
  7) toolchain/iccifort/2019.1.144-GCC-8.2.0-2.31.1
  8) mpi/impi/2018.4.274-iccifort-2019.1.144-GCC-8.2.0-2.31.1
  9) toolchain/iimpi/2019a
 10) numlib/imkl/2019.1.144-iimpi-2019a
 11) toolchain/intel/2019a
```

You notice that Intel MKL is now loaded.

Read the `INSTALL` file under `src/hpl-2.3`. In particular, you'll have to edit and adapt a new makefile `Make.intel64` (inspired from `setup/Make.Linux_Intel64` typically) and provided to you provided to you on [Github](https://raw.githubusercontent.com/ULHPC/tutorials/devel/parallel/mpi/HPL/src/hpl-2.3/Make.intel64) for that purpose.

```bash
$> cd src/hpl-2.3
$> cp ../Make.intel64 .
# OR (if the above command fails)
$> cp ~/git/github.com/ULHPC/tutorials/parallel/mpi/HPL/src/Make.intel64  Make.intel64
# Automatically adapt at least the TOPdir variable to the current directory $(pwd),
# thus it SHOULD be run from 'src/hpl-2.3'
$> sed -i \
   -e "s#^[[:space:]]*TOPdir[[:space:]]*=[[:space:]]*.*#TOPdir = $(pwd)#" \
   Make.intel64
# Check the difference:
$> diff -ru ../Make.intel64 Make.intel64
--- ../Make.intel64     2019-11-19 23:43:26.668794000 +0100
+++ Make.intel64        2019-11-20 00:33:21.077914972 +0100
@@ -68,7 +68,7 @@
 # - HPL Directory Structure / HPL library ------------------------------
 # ----------------------------------------------------------------------
 #
-TOPdir       = $(HOME)/benchmarks/HPL/src/hpl-2.3
+TOPdir = /home/users/svarrette/tutorials/HPL/src/hpl-2.3
 INCdir       = $(TOPdir)/include
 BINdir       = $(TOPdir)/bin/$(ARCH)
 LIBdir       = $(TOPdir)/lib/$(ARCH)
```

In general, to build HPL, you **first** need to configure correctly the file `Make.intel64`.
Take your favorite editor (`vim`, `nano`, etc.) to modify it. In particular, you should adapt:

* `TOPdir` to point to the directory holding the HPL sources (_i.e._ where you uncompress them: ` $(HOME)/tutorials/HPL/src/hpl-2.3`)
    - this was done using the above `sed` command
* Adapt the `MP*` variables to point to the appropriate MPI libraries path.
* Correct the OpenMP definitions `OMP_DEFS`
* (eventually) adapt the `CCFLAGS`
   - in particular, with the Intel compiling suite, you **SHOULD** at least add `-xHost` to ensure the compilation that will auto-magically use the appropriate compilation flags -- see (again) the [Intel Math Kernel Library Link Line Advisor](https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor)
* (eventually) adapt the `ARCH` variable

Here is for instance a suggested difference for intel MPI:

```diff
--- setup/Make.Linux_Intel64    1970-01-01 06:00:00.000000000 +0100
+++ Make.intel64        2019-11-20 00:15:11.938815000 +0100
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
+TOPdir       = $(HOME)/tutorials/HPL/src/hpl-2.3
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
@@ -177,9 +178,9 @@
 #
 CC       = mpiicc
 CCNOOPT  = $(HPL_DEFS)
-OMP_DEFS = -openmp
-CCFLAGS  = $(HPL_DEFS) -O3 -w -ansi-alias -i-static -z noexecstack -z relro -z now -nocompchk -Wall
-#
+OMP_DEFS = -qopenmp
+CCFLAGS  = $(HPL_DEFS) -O3 -w -ansi-alias -i-static -z noexecstack -z relro -z now -nocompchk -Wall -xHost
+
 #
 # On some platforms,  it is necessary  to use the Fortran linker to find
 # the Fortran internals used in the BLAS library.
```

Once tweaked, run the compilation by:

```bash
$> make arch=intel64 clean_arch_all
$> make arch=intel64
```

If you don't succeed by yourself, use the following [Make.intel64](https://raw.githubusercontent.com/ULHPC/tutorials/devel/parallel/mpi/HPL/src/hpl-2.3/Make.intel64).


Once compiled, ensure you are able to run it (you will need at least 4 MPI processes -- for instance with `-N 2 --ntasks-per-node 2`):

```bash
$> cd ~/tutorials/HPL/src/hpl-2.3/bin/intel64
$> cat HPL.dat      # Default (dummy) HPL.dat  input file

# On Slurm cluster (iris), store the output logs into a text file -- see tee
$> srun -n $SLURM_NTASKS ./xhpl | tee test_run.logs
```

Check the output results with `less test_run.logs`. You can also quickly see the 10 best results obtained by using:

``` bash
# ================================================================================
# T/V             N      NB    P     Q               Time             Gflops
# --------------------------------------------------------------------------------
$> grep WR test_run.logs | sort -k 7 -n -r | head -n 10
WR00L2L2          29     3     4     1               0.00             9.9834e-03
WR00L2R2          35     2     4     1               0.00             9.9808e-03
WR00R2R4          35     2     4     1               0.00             9.9512e-03
WR00L2C2          30     2     1     4               0.00             9.9436e-03
WR00R2C2          35     2     4     1               0.00             9.9411e-03
WR00R2R2          35     2     4     1               0.00             9.9349e-03
WR00R2R2          30     2     1     4               0.00             9.8879e-03
WR00R2C4          30     2     1     4               0.00             9.8771e-03
WR00C2R2          35     2     4     1               0.00             9.8323e-03
WR00L2C4          29     3     4     1               0.00             9.8049e-03
```

_Alternatively_, you can use the building script `scripts/build.HPL` to build the HPL sources on both broadwell and skylake nodes (with the corresponding architectures):

``` bash
# (eventually) release you past interactive job to return on access
$> exit

$> cd ~/tutorials/HPL
# Create symlink to the scripts directory
$> ln -s ref.ulhpc.d/scripts .
# Create a logs/ directory to store the Slurm logs
$> mkdir logs

# Now submit two building jobs targeting both CPU architecture
$> sbatch -C broadwell ./scripts/build.HPL -n broadwell  # Will produce bin/xhpl_broadwell
$> sbatch -C skylake   ./scripts/build.HPL -n skylake    # Will produce bin/xhpl_skylake
```

--------------------------
## Preparing batch runs ##

We are now going to prepare launcher scripts to permit passive runs (typically in the `{default | batch}` queue).
We will place them in a separate directory (`runs/`) as it will host the outcomes of the executions on the UL HPC platform .

```bash
$> cd ~/tutorials/HPL
$> mkdir -p runs/{broadwell,skylake}/{1N,2N}/{MPI,Hybrid}/    # Prepare the specific run directory
$> cp ref.ulhpc.d/
```

We are indeed going to run HPL in two different contexts:

1. __Full MPI, with 1 MPI process per (physical) core reserved__.
    - As mentioned in the basics [Parallel computations with OpenMP/MPI](../../basics/) tutorial, it means that you'll typically reserve the nodes using the `-N <#nodes> --ntasks-per-node 28` options for Slurm as there are in general 28 cores per nodes on `iris`.
2. __Hybrid OpenMP+MPI, with 1 MPI process per CPU socket, and as many OpenMP threads as per (physical) core reserved__.
    - As mentioned in the basics [Parallel computations with OpenMP/MPI](../../basics/) tutorial, it means that you'll typically reserve the nodes using the `-N <#nodes> --ntasks-per-node 2 --ntasks-per-socket 1 -c 14` options for Slurm there are in general 2 processors (each with 14 cores) per nodes on `iris`

These two contexts will directly affect the values for the HPL parameters `P` and `Q` since their product should match the total number of MPI processes.


-------------------------
## HPL main parameters ##

Running HPL depends on a configuration file `HPL.dat` -- an example is provided in the building directory i.e. `src/hpl-2.3/bin/intel64/HPL.dat`.

``` bash
$> cat src/hpl-2.3/bin/intel64/HPL.dat
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
4            # of problems sizes (N)
29 30 34 35  Ns
4            # of NBs
1 2 3 4      NBs
0            PMAP process mapping (0=Row-,1=Column-major)
3            # of process grids (P x Q)
2 1 4        Ps
2 4 1        Qs
16.0         threshold
3            # of panel fact
0 1 2        PFACTs (0=left, 1=Crout, 2=Right)
2            # of recursive stopping criterium
2 4          NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
3            # of recursive panel fact.
0 1 2        RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
0            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
0            DEPTHs (>=0)
2            SWAP (0=bin-exch,1=long,2=mix)
64           swapping threshold
0            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
1            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
```

See <http://www.netlib.org/benchmark/hpl/tuning.html> for a description of this file and its parameters (see also the authors [tips](https://www.netlib.org/benchmark/hpl/tuning.html#tips)).

You can use the following sites for finding the appropriate values:

* [Tweak HPL parameters](http://www.advancedclustering.com/act_kb/tune-hpl-dat-file/)
* [HPL Calculator](http://hpl-calculator.sourceforge.net/) to find good parameters
and expected performances


The main parameters to play with for optimizing the HPL runs are:

* `NB`: depends on the CPU architecture, use the recommended blocking sizes (`NB` in HPL.dat) listed after loading the `toolchain/intel` module under  `$EBROOTIMKL/compilers_and_libraries/linux/mkl/benchmarks/mp_linpack/readme.txt`, i.e
     - `NB=192` for the **broadwell** processors available on `iris`
     - `NB=384` on the **skylake** processors available on `iris`
* `P` and `Q`, knowing that the product  `P x Q` **SHOULD** typically be equal to the number of MPI processes.
* Of course `N` the problem size.

An example of P by Q partitioning of a HPL matrix in 6 processes (2x3 decomposition) ([Source](https://www.researchgate.net/profile/Jesus_Labarta/publication/228524393/figure/fig1/AS:301996025368584@1449012871620/An-example-of-P-by-Q-partitioning-of-a-HPL-matrix-in-6-processes-2x3-decomposition.png) )

![](https://www.researchgate.net/profile/Jesus_Labarta/publication/228524393/figure/fig1/AS:301996025368584@1449012871620/An-example-of-P-by-Q-partitioning-of-a-HPL-matrix-in-6-processes-2x3-decomposition.png)

In order to find out the best performance of your system, the largest problem size fitting in memory is what you should aim for.
Since HPL performs computation on an N x N array of Double Precision (DP) elements, and that each double precision element requires `sizeof(double)` = 8 bytes, the memory consumed for a problem size of N is $8N^2$.

It follows that `N` can be derived from a simple dimensional analysis based on the involved volatile memory to compute the number of Double Precision :

$$
N \simeq \alpha\sqrt{\frac{\text{Total Memory Size in bytes}}{\mathtt{sizeof(double)}}} = \alpha\sqrt{\frac{\#Nodes \times RAMsize (GiB) \times 1024^3 }{\mathtt{sizeof(double)}}}
$$

where $\alpha$ is a global ratio normally set to (at least) 80% (best results are typically obtained with $\alpha > 92$%).

Alternatively, one can target a ratio $\beta$ of the total memory used (for instance 85%), i.e.

$$
N \simeq \sqrt{\beta\times\frac{\text{Total Memory Size in bytes}}{\mathtt{sizeof(double)}}}
$$

Note that the two ratios you might consider are of course linked, i.e. $\beta = \alpha^2$

Finally, the problem size should be ideally set to a multiple of the block size `NB`.

**Example of HPL parameters** we are going to try (when using regular nodes on the `batch` partition) are proposed on the below table. Note that we will use _on purpose_ a relatively low value for the ratio  $\alpha$ (or $\beta$), and thus N, to ensure relative fast runs within the time of this tutorial.

| Architecture | #Node | Mode   | MPI proc. |  NB | PxQ                   | $\alpha$ |     N |
|--------------|-------|--------|-----------|-----|-----------------------|----------|-------|
| broadwell    |     1 | MPI    |        28 | 192 | 1x28, 2x14, 4x7       |      0.3 | 39360 |
| broadwell    |     2 | MPI    |        56 | 192 | 1x56, 2x28, 4x14, 7x8 |      0.3 | 55680 |
| broadwell    |     1 | Hybrid |         2 | 192 | 1x2                   |      0.3 | 39360 |
| broadwell    |     2 | Hybrid |         4 | 192 | 1x2                   |      0.3 | 55680 |
|              |       |        |           |     |                       |          |       |
| skylake      |     1 | MPI    |        28 | 384 | 1x28, 2x14, 4x7       |      0.3 | 39168 |
| skylake      |     2 | MPI    |        56 | 384 | 1x56, 2x28, 4x14, 7x8 |      0.3 | 55680 |
| skylake      |     1 | Hybrid |         2 | 384 | 1x2                   |      0.3 | 39168 |
| skylake      |     2 | Hybrid |         4 | 384 | 1x2                   |      0.3 | 55680 |

You can use the script `scripts/compute_N` to compute the value of N depending on the global ratio $\alpha$ (using `-r <alpha>`) or $\beta$ (using `-p <beta*100>`).

``` bash
./scripts/compute_N -h
# 1 Broadwell node, alpha = 0.3
./scripts/compute_N -m 128 -NB 192 -r 0.3 -N 1
# 2 Skylake (regular) nodes, alpha = 0.3
./scripts/compute_N -m 128 -NB 384 -r 0.3 -N 2
# 4 bigmem (skylake) nodes, beta = 0.85
./scripts/compute_N -m 3072 -NB 384 -p 85 -N 4
```

Using the above values, create the appropriate `HPL.dat` files for each case, under the appropriate directory, i.e. `runs/<arch>/<N>N/`


--------------------------------
## Slurm launcher (Intel MPI) ##

Copy and adapt the [default MPI SLURM launcher](https://github.com/ULHPC/launcher-scripts/blob/devel/slurm/launcher.default.sh) you should have a copy in `~/git/ULHPC/launcher-scripts/slurm/launcher.default.sh`

Copy and adapt the [default SLURM launcher](https://github.com/ULHPC/launcher-scripts/blob/devel/slurm/launcher.default.sh) you should have a copy in `~/git/ULHPC/launcher-scripts/slurm/launcher.default.sh`

```bash
$> cd ~/tutorials/HPL/runs
# Prepare a laucnher for intel suit
$> cp ~/git/github.com/ULHPC/launcher-scripts/slurm/launcher.default.sh launcher-HPL.intel.sh
```

Take your favorite editor (`vim`, `nano`, etc.) to modify it according to your needs.

Here is for instance a suggested difference for intel MPI (adapt accordingly):

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
+APPDIR="$HOME/tutorials/HPL/src/hpl-2.3/bin/intel64"
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

Now you should create an input `HPL.dat` file within the `runs/<arch>/<N>N/<mode>`.

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
