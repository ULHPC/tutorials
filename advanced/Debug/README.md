-*- mode: markdown; mode: auto-fill; fill-column: 80 -*-

`README.md`

Copyright (c) 2013 Sebastien Varrette <Sebastien.Varrette@uni.lu>

        Time-stamp: <Tue 2015-11-03 14:42 svarrette>

-------------------


# UL HPC Tutorial:  Know Your Bugs: Weapons for Efficient Debugging

The objective of this tutorial is to review the main tools that can be used to
debug your [parallel] programs. 

* [Slides](slides-debug.pdf)

## Practical session

### 0 - Pre-requisites

Reserve 1 core (for 3h) over the UL HPC platform

      $> ssh gaia-cluster    # OR chaos-cluster
	  $> oarsub -I -l core=1,walltime="03:00:00"


### 1 - GDB Tutorial

Tutorial from [A GDB Tutorial with Examples](http://www.cprogramming.com/gdb.html)

You'll need to load the latest GDB module:

      $> module spider gdb
      $> module load  debugger/GDB


### 2 - Valgrind Tutorial

Tutorial from [Using Valgrind to Find Memory Leaks and Invalid Memory Use](http://www.cprogramming.com/debugging/valgrind.html)

You'll also need to load the appropriate module

      $> module spider valgrind
      $> module load debugger/Valgrind


## 3 - Bug Hunting

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
  
