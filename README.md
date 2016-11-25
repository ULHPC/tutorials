-*- mode: markdown; mode: visual-line;  -*-

       Time-stamp: <Wed 2015-06-10 10:30 svarrette>

-------------------

# UL HPC Tutorials 

[![License](http://img.shields.io/:license-GPL3.0-blue.svg)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](https://readthedocs.org/projects/ulhpc-tutorials/?badge=latest)
[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](http://hpc.uni.lu)

      Copyright (c) 2013-2015 UL HPC Team aka. S. Varrette, H. Cartiaux, V. Plugaru, S. Diehl <hpc-sysadmins@uni.lu>

| [Project Page](https://github.com/ULHPC/tutorials) |  [Documentation](https://ulhpc-tutorials.readthedocs.org/en/latest/) | [Issues](https://github.com/ULHPC/tutorials/issues) |

## Synopsis

This repository holds a set of tutorials to help the users of the [UL HPC](https://hpc.uni.lu) platform to better understand or simply use our platform. 

## Tutorials layout

For each subject, you will have a dedicated directory organized as follows:

* `README.md`: hold the description of the tutorial
* (eventually) `Makefile`: a [GNU make](http://www.gnu.org/software/make/) configuration file, consistent with the following conventions for the lazy users who do not wish to do all the steps proposed in the tutorial: 
  
   - `make fetch` will retrieve whatever archives / sources are required to
    perform the tutorial
   - `make build` will automate the build of the software
   - `make run_interactive` to perform a sample run in interactive mode
   - `make run` to perform a sample run in passive mode (_i.e_  via `oarsub -S ...` typically)
   - (eventually) `make plot` to draw a plot illustrating the results obtained through the run.

## List of proposed tutorials

The organization and indexing of the tutorials follows the program of the last [UL HPC School](http://hpc.uni.lu/hpc-school/) and thus is subject to changes over time. 
 
__Basic tutorials__:

* PS1B: [Getting Started on the UL HPC platform (SSH, data transfer, OAR, modules, monitoring)](/basic/getting_started)
* PS2A: [HPC workflow with sequential jobs](/basic/sequential_jobs)

__MPI__

* PS3A-1: [running the OSU Micro-Benchmarks](/advanced/OSU_MicroBenchmarks)
* PS3A-2: [running HPL](/advanced/HPL)

__Mathematics__:

* PS3B: [running MATLAB](/advanced/MATLAB1)
* PS3C: [running R](/advanced/R)

__Advanced Software Management__

* PS 4A: [Software environment generation: RESIF/Easybuild](/advanced/RESIF/)

__Bio-informatic__

* PS 6A: [Running Bio-informatics software: test cases on Abyss, GROMACS, Bowtie2/TopHat, mpiBLAST](/advanced/Bioinformatics/)

__Parallel Debuggers__

* [Direct, Reverse and parallel Memory debugging with TotalView](/advanced/TotalView)
* [Allinea](/advanced/Allinea)

## Issues / Feature request

You can submit bug / issues / feature request using the [ULHPC/tutorials Tracker](https://github.com/ULHPC/tutorials/issues). 

## Developments / Contributing to the code 

If you want to contribute to the code, you shall be aware of the way this module is organized. 
These elements are detailed on [`docs/contributing.md`](contributing.md).

You are more than welcome to contribute to its development by [sending a pull request](https://help.github.com/articles/using-pull-requests). 

## Online Documentation

[Read the Docs](https://readthedocs.org/) aka RTFD hosts documentation for the open source community and the [ULHPC/sysadmins](https://github.com/ULHPC/tutorials) has its documentation (see the `docs/` directly) hosted on [readthedocs](http://ulhpc-tutorials.rtfd.org).

See [`docs/rtfd.md`](rtfd.md) for more details.

## Licence

This project and the sources proposed within this repository are released under the terms of the [GPL-3.0](LICENCE) licence.

[![Licence](https://www.gnu.org/graphics/gplv3-88x31.png)](LICENSE)
