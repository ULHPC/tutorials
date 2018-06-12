[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/advanced/Debug/slides-debug.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/advanced/Debug/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/advanced/Debug/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Know Your Bugs: Weapons for Efficient Debugging

      Copyright (c) 2013-2018 X. Besseron, UL HPC Team <hpc-sysadmins@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/advanced/Debug/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/advanced/Debug/slides-debug.pdf)

The objective of this tutorial is to review the main tools that can be used to
debug your [parallel] programs.

## Hands/On 0 - Pre-requisites

Reserve 1 core (for 3h) over the UL HPC platform

      $> ssh gaia-cluster    # OR chaos-cluster
	  $> oarsub -I -l core=1,walltime="03:00:00"


## Hands/On 1 - GDB Tutorial

Tutorial from [A GDB Tutorial with Examples](http://www.cprogramming.com/gdb.html)

You'll need to load the latest GDB module:

      $> module spider gdb
      $> module load  debugger/GDB


## Hands/On 2 - Valgrind Tutorial

Tutorial from [Using Valgrind to Find Memory Leaks and Invalid Memory Use](http://www.cprogramming.com/debugging/valgrind.html)

You'll also need to load the appropriate module

      $> module spider valgrind
      $> module load debugger/Valgrind


## Hands/On 3 - Bug Hunting

A list of programs demonstrating the different kind of bus are available in the `exercises` directory.
Try the different debugging tools on every example to see how they behave and find the bugs.

Run the following command to download all the exercises:

```
$> git clone https://github.com/ULHPC/tutorials.git ulhpc-tutorials
$> cd ulhpc-tutorials/advanced/Debug/exercises/
```

*Notes*:

* You can compile each program manually using `gcc` or `icc` (the latest coming from the `toolchains/ictce` module). You are encouraged to try both to see how differently they behave. Example: `gcc program.c -o program`. Add any additional parameter you might need.
* Some program required additional options to be compiled. They are indicated in comment at the beginning of each source file.
