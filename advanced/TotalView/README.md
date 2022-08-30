-*- mode: markdown; mode: auto-fill; fill-column: 80 -*-

Copyright (c) 2014 Sebastien Varrette <Sebastien.Varrette@uni.lu>

-----------------------------------------------------------------
# UL HPC Tutorial: Direct, Reverse and parallel Memory debugging with TotalView

[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/advanced/TotalView/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/advanced/TotalView/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/advanced/TotalView/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

[![](https://github.com/ULHPC/tutorials/raw/devel/advanced/TotalView/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/advanced/TotalView/TV_TrainingLab_Manual20130603_EN.pdf)


The objective of this tutorial is to get a brief overview of the [TotalView](http://www.roguewave.com/products/totalview.aspx), a GUI-based source code defect analysis tool that gives you unprecedented control over processes and thread execution and visibility into program state and variables.

This practical session will be organized as follows:

* Startup and Overview
* IHM Navigation and process control
* Action points (TP)
* Examination and Data Analysis (TP)
* Debugging Parallel Applications (TP)
* Memory reports using MemoryScape (TP)
* Remote Debugging
* CUDA Debugging
* Xeon Phi Debugging
* Memory Debugging with MemoryScape
* Events and memory errors (TP)
* Delayed Scripted debugging (non-interactive) (TP)
* Reverse debugging using ReplayEngine (TP)
* Asynchronous control of Parallel Applications (TP)
* Type Transformation (TP)

While TotalView is available on the [UL HPC Platform](https://hpc.uni.lu), the specific exercises proposed have been embedded into a Virtual Machine (VM) you will need to setup to run this practical session.

## Pre-Requisites

* Install [Oracle's VirtualBox](http://www.virtualbox.org/)
* Download the bootable ISO [`tv_trainingcdU10_20140815.iso`](https://hpc.uni.lu/download/tv_trainingcdU10_20140815.iso) (md5sum: *30a844ddda80ddf505c28eb4f4b6f1bf*) which contains:
  * a bootable Linux Ubuntu distribition
  * a version of TotalView
  * the PDF documents describing the practical session.

If the participants did not have the time to download the ISO, they shall come with a USB stick having a capacity of at least 2GB.

To create a new VM:

* Open VirtualBox
* Select the `New` icon
  * Name: `TotalView`
  * Type: `Linux`
  * Version: `Ubuntu 64 bits`
  * Memory Size: `512MB`
  * `Create a virtual hard drive now`

  Click on "Create": select a VDI format, dynamically allocated.
* Now select the `Start` icon over the newly created VM
  * open the folder icon to browse your disk and select the  `tv_trainingcdU10_20140815.iso` ISO on which you'll boot

 You're now ready for the tutorial.


## Tutorial


- [Content of the tutorial](https://github.com/ULHPC/tutorials/blob/devel/advanced/TotalView/TV_TrainingLab_Manual20130603_EN.pdf?raw=true)
- [Answers of the tutorial](https://github.com/ULHPC/tutorials/blob/devel/advanced/TotalView/TV_TrainingLab_Answers20130603_EN.pdf?raw=true)
