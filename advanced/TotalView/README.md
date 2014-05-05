-*- mode: markdown; mode: auto-fill; fill-column: 80 -*-

`README.md`

Copyright (c) 2014 Sebastien Varrette <Sebastien.Varrette@uni.lu>

        Time-stamp: <Mar 2014-05-01 15:09 svarrette>

-------------------


# UL HPC Tutorial: Direct, Reverse and parallel Memory debugging with TotalView

The objective of this tutorial is to get a brief overview of the [TotalView](http://www.roguewave.com/products/totalview.aspx), a GUI-based source code defect analysis tool that gives you unprecedented control over processes and thread execution and visibility into program state and variables.

This practical session will be organized as follows:

* Startup and Overview
* IHM Navigation and process control
* Action points (TP)
* Examination and Data Analysis (TP)
* Debugging Parallel Applications (TP)
* Memory reports using MemoryScape (TP)* Remote Debugging
* CUDA Debugging
* Xeon Phi Debugging
* Memory Debugging with MemoryScape
* Events and memory errors (TP)
* Delayed Scripted debugging (non-interactive) (TP)
* Reverse debugging using ReplayEngine (TP)
* Asynchronous control of Parallel Applications (TP)
* Type Transformation (TP)

While TotalView is available on the [UL HPC Platform](http://hpc.uni.lu), the specific exercises proposed have been embedded into a Virtual Machine (VM) you will need to setup to run this practical session. 

## Pre-Requisites

* Install [Oracle's VirtualBox](http://www.virtualbox.org/)
* Download the bootable ISO `tv_trainingcdU10_20140815.iso` (md5sum: *30a844ddda80ddf505c28eb4f4b6f1bf*) which contains: 
  * a bootable Linux Ubuntu distribition
  * a version of TotalView
  * the PDF documents describing the practical session.

If the participants did not have the time to download the ISO, they shall come with a USB stick having a capacity of at least 2GB.
