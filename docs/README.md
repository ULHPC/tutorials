[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Uni.lu High Performance Computing (HPC) Tutorials

     Copyright (c) 2013-2021 UL HPC Team <hpc-team@uni.lu>

This repository holds a set of tutorials to help the users of the [UL HPC](https://hpc.uni.lu) platform to better understand or simply use our platform.

This is the main page of the documentation for this repository, which relies on [MkDocs](http://www.mkdocs.org/) and the [Read the Docs](http://readthedocs.io) theme.
In particular, the latest version of these tutorials is available online:

<http://ulhpc-tutorials.rtfd.io>

The list of proposed tutorials is continuously evolving.
They used on a regular basis during the [UL HPC School](https://hpc.uni.lu/education/hpcschool) we organise.

[![ULHPC School](https://img.shields.io/badge/event-ULHPC--School--2021-green)](hpc-school.md)

So far, the following tutorials are proposed:

| **Category**            | **Description**                                                                                       | **Level**      |
| :----------:            | ----------------------------------------------------------------------------                          | -------------- |
|                         | [Pre-requisites and Setup instructions](setup/preliminaries/)                                         | beginners      |
| _Basic_                 | [Introduction to UNIX/Linux Shell and Command lines](linux-shell/)                                    | beginners      |
|                         | [Getting Started on the UL HPC platform](beginners/)                                                  | beginners      |
|                         | [Monitoring & Profiling I: why, what, how, where to look](basic/monitoring/)                          | beginners      |
| _Emb. Parallel Jobs_    | [GNU Parallel](sequential/manytasks-manynodes2/)                                                      | beginners      |
|                         | [HPC Management of Sequential and Embarrassingly Parallel Jobs](sequential/basics/)                   | beginners      |
| _Scheduling_            | [Advanced scheduling with SLURM](scheduling/advanced/)                                                | intermediate   |
| _Software Management_   | [HPC Software Building: optimizing and complementing the ULHPC software set](tools/easybuild/)        | beginners      |
|                         | [The Spack package manager for supercomputers](tools/spack/)                                          | beginners      |
| _Data Management_       | [Data Management on UL HPC Facility](data/)                                                           | beginners      |
| _Debuging & Profiling_  | [Know Your Bugs: Weapons for Efficient Debugging](debugging/basics/)                                  | intermediate   |
|                         | [Advanced debugging on the UL HPC platform](debugging/advanced/)                                      | intermediate   |
|                         | (OLD) [Unified profiling and debugging with Allinea](advanced/Allinea/)                               | intermediate   |
|                         | (OLD) [Direct,  Reverse and parallel Memory debugging with TotalView](advanced/TotalView/)            | intermediate   |
| _MPI_                   | [Scalable Science and Parallel computations with OpenMP/MPI](parallel/basics/)                        | intermediate   |
|                         | [OSU Micro-Benchmarks](parallel/mpi/OSU_MicroBenchmarks/)                                             | intermediate   |
|                         | [High-Performance Linpack (HPL) benchmarking on UL HPC platform](parallel/mpi/HPL/)                   | intermediate   |
|                         | [HPCG benchmarking on UL HPC platform](parallel/hybrid/HPCG/)                                         | intermediate   |
| _Python_                | [Prototyping with Python](python/basics/)                                                             | beginners      |
|                         | [Python: Use Jupyter notebook on UL HPC](python/advanced/jupyter)                                     | intermediate   |
|                         | [Use Python Celery on UL HPC](python/advanced/celery/)                                                | advanced       |
|                         | [Scalable computing with Dask](python/advanced/dask-ml/)                                              | advanced       |
|                         | [Parallel machine learning with scikit-learn](python/advanced/scikit-learn)                           | intermediate   |
|                         | [Parallel evolutionary computing with Scoop/Deap](python/advanced/scoop-deap)                         | intermediate   |
| _Mathematics_           | [MATLAB (interactive, passive and sequential jobs)](maths/matlab/basics/)                             | intermediate   |
|                         | [Advanced MATLAB execution: checkpointing and parallel jobs](maths/matlab/advanced/)                  | advanced       |
|                         | [R / Statictical Computing](maths/R/)                                                                 | intermediate   |
|                         | [Cplex / Gurobi](maths/Cplex-Gurobi/)                                                                 | intermediate   |
| _Bioinformatics_        | [Bioinformatics software on the UL HPC platform](bio/basics/)                                         | intermediate   |
|                         | [Galaxy Introduction Exercise: From Peaks to Genes](bio/galaxy/)                                      | intermediate   |
| _CFD/MD/Chemistry_      | [Scalable Science II (Advanced - Computational Physics, Chemistry & Engineering apps)](multiphysics/) | advanced       |
| _Big Data_              | [Running Big Data Application using Apache Hadoop and Spark ](bigdata/)                               | intermediate   |
| _Machine/Deep Learning_ | [Machine and Deep learning workflows](deep_learning/)                                                 | intermediate   |
|                         | [Distributed Deep Learning with Horovod](deep_learning/horovod/)                                      | intermediate   |
| _Containers_            | [Singularity](containers/singularity/)                                                                | advanced       |
|                         | [Singularity with Infiniband](containers/singularity-inf/)                                            | advanced       |
|                         | [Reproducibility and HPC Containers ](containers/ULHPC-containers/)                                   | advanced       |
| _Virtualization_        | (OLD) [Create and reproduce work environments using Vagrant](advanced/Vagrant/)                       | intermediate   |
|                         | [Deploying virtual machines with Vm5k on Grid'5000](advanced/vm5k/)                                   | intermediate   |
| _GPU Programming_       | [Introduction to GPU programming with CUDA (C/C++)](cuda/)                                            | intermediate   |
|                         | [Image Convolution with GPU and CUDA](cuda/exercises/convolution/)                                    | advanced       |
|                         | [Introduction to OpenCL](gpu/opencl/)                                                                 | intermediate   |
|                         | [Introduction to OpenACC Programming Model (C/C++ and Fortran)](gpu/openacc/basics/)                  | intermediate   |
|                         | [Solving the Laplace Equation on GPU with OpenAcc](gpu/openacc/laplace/)                              | advanced       |
| _IPU Programming_       | [Tensorflow on Graphcore IPU](experimental_hardware/graphcore_ipu/machine_learning/)                  | intermediate   |
<!-- |                         | [C Poplar on IPU](experimental_hardware/graphcore_ipu/c_poplar/)                                      | intermediate   | -->
| _Misc_                  | [Reproducible Research at the Cloud Era](misc/reproducible-research/)                                 | intermediate   |

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
