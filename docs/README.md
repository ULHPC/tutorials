-*- mode: markdown; mode: visual-line;  -*-

# UL HPC Tutorials

[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/basic/getting_started/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/basic/getting_started/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

      Copyright (c) 2015-2017 UL HPC Team <hpc-sysadmins@uni.lu>
      .                       aka. S. Varrette, H. Cartiaux, V. Plugaru, S. Diehl and C.Parisot

This repository holds a set of tutorials to help the users of the [UL HPC](https://hpc.uni.lu) platform to better understand or simply use our platform.
The list of proposed tutorials are evolving and are used on a regular basis during the [UL HPC School](http://hpc.uni.lu/hpc-school/) we organise.

So far, the following tutorials are proposed:

| **Category**          | **Description**                                                                 | **Tags**                                    | **Level**      |
| :----------:          | ----------------------------------------------------------------------------    | :--------------:                            | -------------- |
| _Basic_               | [Getting Started on the UL HPC platform](basic/getting_started/)                | ssh, oar, slurm, screen                     | beginners      |
|                       | [HPC workflow with sequential jobs](basic/sequential_jobs/)                     | ssh, gromacs, python, java                  | beginners      |
|                       | [Know Your Bugs: Weapons for Efficient Debugging](advanced/Debug/)              | debug                                       | intermediate   |
| _Software Management_ | [Building [custom] software with EasyBuild](advanced/EasyBuild/)                | easybuild                                   | beginners      |
| _MPI_                 | [UL HPC MPI Tutorial: OSU Micro-Benchmarks](advanced/OSU_MicroBenchmarks/)      | mpi, perfs                                  | intermediate   |
|                       | [High-Performance Linpack (HPL) benchmarking on UL HPC platform](advanced/HPL/) | mpi, perfs                                  | intermediate   |
|                       | [HPCG benchmarking on UL HPC platform](advanced/HPCG/)                          | mpi, perfs                                  | intermediate   |
| _Mathematics_         | [MATLAB execution on the UL HPC platform](advanced/MATLAB1/)                    | maths                                       | intermediate   |
|                       | [R / Statictical Computing](advanced/R/)                                        | maths, R                                    | intermediate   |
| _Bioinformatics_      | [Bioinformatics software on the UL HPC platform](advanced/Bioinformatics/)      | bio, abyss, gromacs, bowtie2, mpiblast      | intermediate   |
|                       | Galaxy Introduction Exercise: From Peaks to Genes                               | bio                                         | intermediate   |
| _Parallel Debuging_   | Unified profiling and debugging with Allinea                                    | debug                                       | intermediate   |
|                       | Direct,  Reverse and parallel Memory debugging with TotalView                   | debug                                       | intermediate   |
| _Virtualization_      | Create and reproduce work environments using Vagrant                            | vm, vagrant                                 | intermediate   |
|                       | Deploying virtual machines with Vm5k on Grid'5000                               | vm, g5k                                     | intermediate   |
| _CFD/MD/Chemistry_    | Running parallel software: test cases on CFD / MD / Chemistry applications      | mpi, OpenFoam, NAMD, ASE, Abinit, QExpresso | advanced       |
| _Big Data_            | Running Big Data Application using Apache Spark on UL HPC platform              | bigdata, spark                              | intermediate   |
| _Misc_                | Advanced workflows on sequential jobs management                                | oar, fault-tolerance                        | advanced       |
|                       | Advanced scheduling with SLURM                                                  | slurm                                       | intermediate   |
|                       | [Advanced] Prototyping with Python                                              | python,scoop,pythran                        | advanced       |
|                       |                                                                                 |                                             |                |


## Issues / Feature request

You can submit bug / issues / feature request using the [ULHPC/tutorials Tracker](https://github.com/ULHPC/tutorials/issues).

## Developments / Contributing to the code

If you want to contribute to the code, you shall be aware of the way this module is organized.
These elements are detailed on [`docs/contributing.md`](contributing.md).

You are more than welcome to contribute to its development by [sending a pull request](https://help.github.com/articles/using-pull-requests).

### Online Documentation

[Read the Docs](https://readthedocs.org/) aka RTFD hosts documentation for the open source community and the [ULHPC/sysadmins](https://github.com/ULHPC/tutorials) has its documentation (see the `docs/` directly) hosted on [readthedocs](http://ulhpc-tutorials.rtfd.org).

See [`docs/rtfd.md`](rtfd.md) for more details.

### Licence

This project and the sources proposed within this repository are released under the terms of the [GPL-3.0](LICENCE) licence.

[![Licence](https://www.gnu.org/graphics/gplv3-88x31.png)](LICENSE)
