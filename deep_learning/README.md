[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/deep_learning/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/deep_learning/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/deep_learning/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)


# Machine and Deep Learning on the UL HPC Platform

     Copyright (c) 2013-2019 UL HPC Team  <hpc-sysadmins@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/deep_learning/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/deep_learning/slides.pdf)

The objective of this tutorial is to demonstrate how to build and run on top of the [UL HPC](http://hpc.uni.lu) platform a couple of reference Machine and Deep learning frameworks.

This tutorial is a follow-up (and thus depends on) **two** other tutorials:

* [Easybuild tutorial `tools/easybuild/`](http://ulhpc-tutorials.readthedocs.io/en/latest/tools/easybuild/)
* [Advanced Python tutorial `python/advanced/`](https://ulhpc-tutorials.readthedocs.io/en/latest/python/advanced/)
* [Big Data tutorial](https://ulhpc-tutorials.readthedocs.io/en/latest/bigdata/)

_Reference_:

* [Deep Learning with Apache Spark and TensorFlow](https://databricks.com/blog/2016/01/25/deep-learning-with-apache-spark-and-tensorflow.html)
* [Tensorflow tutorial on MNIST](https://www.tensorflow.org/versions/master/get_started/mnist/beginners)
    - MNIST dataset: see [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/)

--------------------
Outline

* Develop a model with Keras
	- Work on a CPU node
	- Test on a GPU (interactive)
* Launch batch job on a single node
	- run on GPU

--------------------
## Pre-requisites ##

Ensure you are able to [connect to the UL HPC clusters](https://hpc.uni.lu/users/docs/access.html)
**For all tests and compilation, you MUST work on a computing node**

You'll need to prepare the data sources required by this tutorial once connected

This hands-on is **not** about learning Machine Machine Learning (ML)/Deep Learning (DL).
For that, we encourage you to take a look at the **EXCELLENT** courses being taught at as part of [Master Datascience Paris Saclay](http://datascience-x-master-paris-saclay.fr/) and available on

https://github.com/m2dsupsdlclass/lectures-labs

In particular, start with the
[Introduction to Deep Learning](https://m2dsupsdlclass.github.io/lectures-labs/slides/01_intro_to_deep_learning/index.html#1) _by_ Charles Ollion and Olivier Grisel.

----------------------------------
## 1. Preliminary installations ##

### Step 1.a Connect to a cluster node

We only need one core of one node, because this is for configuring the environment.

```bash
$> ssh iris-cluster
$> srun -n1 -c1 -pinteractive --pty bash -i
[]$ scontrol show job $SLURM_JOB_ID  # sj $SLURM_JOB_ID
```

### Step 1.b Prepare python virtualenvs

Our working environment consists of:

- python 3
- CUDA toolkit for the GPU tests
- tensorflow, both for CPU and GPU. 

Tensorflow comes in different versions for CPU and GPU.
We install both, and will select the correct one (matching the reserved node) before running the code.
To do so, we can rely on virtualenv.
If you have never used virtualenv before, please have a look at [Python1 tutorial](http://ulhpc-tutorials.readthedocs.io/en/latest/python/basics/).

```bash
[]$ # Load the required version of python, and the CUDA libraries:
$> module load lang/Python/3.6.4-foss-2018a system/CUDA numlib/cuDNN
$> module list # ml
```
To quickly recover our configuration, when developing interactively, we can save a module configuration:
```bash
$> # Save the module in a specific module list named 'tf': 
$> module save tf  # module s tf
$> module purge
$> module restore tf  # module r tf
```
Set up virtualenv:
```bash
$> # Create a common directory for your python environments:
$> mkdir ~/venv && cd ~/venv
$> # for the CPU tensorflow version
$> virtualenv tfcpu
$> # for the GPU tensorflow version
$> virtualenv tfgpu
```
## Step 1.c. Install Tensorflow

See also [installation notes](https://www.tensorflow.org/install/)

You should have 2 virtualenvs created: tfcpu and tfgpu.

```bash
# ENSURE you are in tfenv environment
$> 
(tfenv) $> pip install tensorflow-gpu
```

To check the installation, within the python interpreter:
```python
>>> import tensorflow as tf
>>> tf.VERSION
>>> tf.keras.__version__
```

-----------------------------------------------------------------
## 2. Interactive development of a TensorFlow tutorial ##

### Pre-requisites

If Python is not already loaded:
```bash
$> module restore tf
```

If your virtual environment is not available, activate it:
```bash
$> source ~/venv/tfenv/bin/activate
(tfenv) $>
```

At the time of writing, recent versions of numpy broke a tensorflow tutorial source file.
In case of exception when unpickling the training data, apply a corrective patch:

```bash
# the file imdb.patch comes from the github.com repository
$> cp imdb.patch $VIRTUAL_ENV  
$> cd $VIRTUAL_ENV
$> patch -p0 <imdb.patch
```

### TensorFlow example

Follow the tutorial on: 
[TensorFlow text classification tutorial](https://www.tensorflow.org/tutorials/keras/basic_text_classification).

This tutorial can be assembled on a regular, CPU-only, node.

-----------------------------------------------------------------
## 3. Batch training of a TensorFlow application ##

One solution to the tutorial can be found here [imdb-train.py](./imdb-train.py).

References:

* [TensorFlow text classification tutorial](https://www.tensorflow.org/tutorials/keras/basic_text_classification)
* [Tensorflow Tutorial](https://www.tensorflow.org/versions/master/get_started/mnist/beginners)


