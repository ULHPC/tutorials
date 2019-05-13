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
Outline 2019

* Develop, with Keras. 
	- code on cpu, later deploy on gpu then
	- env: python foss, add 
	- test on gpu (interactive)
* Launch batch Keras job. 
	- run on gpu.

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

```bash
$> ssh iris-cluster
$> srun -N 1 -c 1 -p interactive --pty bash -i
[<user-name>@iris-<some-node-number>scontrol show job $SLURM_JOB_ID
```

### Step 1.b Prepare python virtualenv

If you have never used virtualenv before, please have a look at [Python1 tutorial](http://ulhpc-tutorials.readthedocs.io/en/latest/python/basics/).

```bash
# Load your prefered version of Python
$> module load lang/Python/2.7.14-foss-2018a
# Create a directory for your environment:
$> mkdir ~/venv && cd ~/venv
$> virtualenv dlenv
$> source dlenv/bin/activate
# deactivate to leave the dlenv environment
```
## Step 1.b. Install Tensorflow

See also [installation notes](https://www.tensorflow.org/install/)

Assuming you work within a `dlenv` virtualenv environment:

```
# ENSURE you are in dlenv environment
$> pip install tensorflow
```
-----------------------------------------------------------------
## 2. Example application ##

### Overview

Keras, why. Overview of APIs.

Development

References:

* [Tensorflow Tutorial](https://www.tensorflow.org/versions/master/get_started/mnist/beginners)

MNIST is a simple computer vision dataset. It consists of images of handwritten digits like these:

![](https://www.tensorflow.org/images/MNIST.png)

The MNIST data is split into three parts:

1. 55,000 data points of training data (`mnist.train`),
2. 10,000 points of test data (`mnist.test`),
3. 5,000 points of validation data (`mnist.validation`).

This split is very important: it's of course essential in ML that we have separate data which we don't learn from so that we can make sure that what we've learned actually generalizes!

Each image is 28 pixels by 28 pixels. We can interpret this as a big array of numbers:

![](https://www.tensorflow.org/images/MNIST-Matrix.png)

Thus after flattening the image into vectors of 28*28=784, we obtain as `mnist.train.images` a tensor (an n-dimensional array) with a shape of [55000, 784].

MNIST images is of a handwritten digit between zero and nine. So there are only ten possible things that a given image can be.
In this hands-on, we will design two **classifiers** for MNIST images:

1. A very [simple MNIST classifier](https://www.tensorflow.org/get_started/mnist/beginners), able to reach an accuracy of around 92% -- see Jupyter notebook `mnist-1-simple.ipynb`
2. A more advanced [deep MNIST classifier using convolutional layers](https://www.tensorflow.org/get_started/mnist/pros), which will reach an accuracy of around 99.25%, which is way better than the previously obtained results (around 92%) -- see Jupyter Notebook `mnist-2-deep_convolutional_NN.ipynb`


