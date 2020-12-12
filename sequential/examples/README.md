[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/sequential/examples/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/sequential/examples/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/sequential/examples) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Past HPC School Exercises: HPC workflow with sequential jobs

    Copyright (c) 2013-2019 UL HPC Team <hpc-sysadmins@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/sequential/examples/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/sequential/examples/slides.pdf)

**Prerequisites**

Make sure you have followed the tutorial ["Getting started"](../../getting-started/).

For many users, the typical usage of the HPC facilities is to execute 1 program with many parameters.
On your local machine, you can just start your program 100 times sequentially.
However, you will obtain better results if you parallelize the executions on a HPC Cluster.

During this session, we will see 3 use cases:

* __[Exercise 1](object_recognition.md)__: Use the serial launcher (1 node, in sequential and parallel mode);
    - application to _Object recognition_ with [Tensorflow](https://www.tensorflow.org/) and [Python Imageai](https://github.com/OlafenwaMoses/ImageAI)
* __[Exercise 2](watermarking.md)__: Use the generic launcher, distribute your executions on several nodes (python script);
    - Illustration on an _Image Watermarking process_ in Python
* __[Exercise 3](jcell.md)__: Advanced use case, using a Java program: "JCell".

We will use the following github repositories:

* [ULHPC/launcher-scripts](https://github.com/ULHPC/launcher-scripts)
    - **UPDATE (Dec 2020)** This repository is **deprecated** and kept for archiving purposes only. Consider the up-to-date launchers listed at the root of the ULHPC/tutorials repository, under `launchers/`
* [ULHPC/tutorials](https://github.com/ULHPC/tutorials)
