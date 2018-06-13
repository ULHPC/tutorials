[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# High Performance Computing (HPC) Tutorials

     Copyright (c) 2013-2018 UL HPC Team <hpc-sysadmins@uni.lu>

This repository holds a set of tutorials to help the users of the [UL HPC](https://hpc.uni.lu) platform to better understand or simply use our platform.

This is the main page of the documentation for this repository, which relies on [MkDocs](http://www.mkdocs.org/) and the [Read the Docs](http://readthedocs.io) theme.
In particular, the latest version of these tutorials is available online:

<http://ulhpc-tutorials.rtfd.io>

The list of proposed tutorials is continuously evolving.
They used on a regular basis during the [UL HPC School](http://hpc.uni.lu/hpc-school/) we organise.
So far, the following tutorials are proposed:

| **Category**           | **Description**                                                                             | **Level**      |
| :----------:           | ----------------------------------------------------------------------------                | -------------- |
| _Basic_                | [Getting Started on the UL HPC platform](beginners/)                                        | beginners      |
|                        | [HPC workflow with sequential jobs](basic/sequential_jobs/)                                 | beginners      |
|                        | [Prototyping with Python](python/basics/)                                                   | beginners      |
|                        | [Advanced prototyping with Python](python/advanced/)                                        | intermediate   |
| _Scheduling_           | [Advanced scheduling with SLURM](scheduling/advanced)                                       | intermediate   |
| _Software Management_  | [Building [custom] software with EasyBuild](tools/easyBuild/)                               | beginners      |
| _Debuging & Profiling_ | [Know Your Bugs: Weapons for Efficient Debugging](debugging/basics/)                        | intermediate   |
|                        | [Advanced debugging on the UL HPC platform](debugging/advanced/)                            | intermediate   |
|                        | (OLD) [Unified profiling and debugging with Allinea](advanced/Allinea/)                     | intermediate   |
|                        | (OLD) [Direct,  Reverse and parallel Memory debugging with TotalView](advanced/TotalView/)  | intermediate   |
| _MPI_                  | [Parallel computations with OpenMP/MPI](parallel/basics/)                                   | intermediate   |
|                        | [UL HPC MPI Tutorial: OSU Micro-Benchmarks](parallel/mpi/OSU_MicroBenchmarks/)              | intermediate   |
|                        | [High-Performance Linpack (HPL) benchmarking on UL HPC platform](parallel/mpi/HPL/)         | intermediate   |
|                        | [HPCG benchmarking on UL HPC platform](parallel/hybrid/HPCG/)                               | intermediate   |
| _Mathematics_          | [MATLAB (interactive, passive and sequential jobs)](maths/matlab/basics/)                   | intermediate   |
|                        | [Advanced MATLAB execution: checkpointing and parallel jobs](maths/matlab/advanced/)        | advanced       |
|                        | [R / Statictical Computing](maths/R/)                                                       | intermediate   |
| _Bioinformatics_       | [Bioinformatics software on the UL HPC platform](bio/basics/)                               | intermediate   |
|                        | [Galaxy Introduction Exercise: From Peaks to Genes](bio/galaxy/)                            | intermediate   |
| _CFD/MD/Chemistry_     | [Running parallel software: test cases on CFD / MD / Chemistry applications](multiphysics/) | advanced       |
| _Big Data_             | [Running Big Data Application using Apache Hadoop and Spark ](bigdata/)                     | intermediate   |
| _Containers_           | [HPC Containers with Singularity](containers/singularity/)                                  | advanced       |
| _Virtualization_       | (OLD) [Create and reproduce work environments using Vagrant](advanced/Vagrant/)             | intermediate   |
|                        | [Deploying virtual machines with Vm5k on Grid'5000](advanced/vm5k/)                         | intermediate   |
| _Misc_                 | [Reproducible Research at the Cloud Era](misc/reproducible-research/)                       | intermediate   |

__List of contributors__

* See [`docs/contacts.md`](contacts.md).
* In the advent where you want to contribute yourself to these tutorials, do not hesitate! See below for instructions.

__Issues / Feature request__

* You can submit bug / issues / feature request using the [`ULHPC/tutorials` Tracker](https://github.com/ULHPC/tutorials/issues).
    - See also [`docs/contributing/`](docs/contributing/) for further information.

__ Contributing__

* If you want to contribute to the code, you shall be aware of the way this module is organized.
* These elements are detailed on [`docs/contributing.md`](contributing.md).
    - You are more than welcome to contribute to its development by [sending a pull request](https://help.github.com/articles/using-pull-requests).

__Online Documentation__

* [Read the Docs](https://readthedocs.org/) aka RTFD hosts documentation for the open source community and the [ULHPC/sysadmins](https://github.com/ULHPC/tutorials) has its documentation (see the `docs/` directly) hosted on [readthedocs](http://ulhpc-tutorials.rtfd.org).
* See [`docs/rtfd.md`](rtfd.md) for more details.

__Licence__

This project and the sources proposed within this repository are released under the terms of the [GPL-3.0](LICENCE) licence.

[![Licence](https://www.gnu.org/graphics/gplv3-88x31.png)](LICENSE)
