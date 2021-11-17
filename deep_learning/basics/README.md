[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/deep_learning/basics/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/deep_learning/basics/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/deep_learning/basics/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)


# Machine and Deep Learning on the UL HPC Platform

     Copyright (c) 2013-2019 UL HPC Team  <hpc-sysadmins@uni.lu>

[![](cover_slides.png)](slides.pdf)

This tutorial demonstrates how to develop and run on the [UL HPC](https://hpc.uni.lu) platform a deep learning application using the Tensorflow framework.

The scope of this tutorial is *single* node execution, multi-CPU and multi-GPU.
Another tutorial covers multi-node execution.

--------------------
## Pre-requisites ##

Ensure you are able to [connect to the UL HPC clusters](https://hpc-docs.uni.lu/connect/access/).
**For all tests and compilation, you MUST work on a computing node**

This hands-on is **not** a tutorial on Deep Learning (DL).
For that, we encourage you to take a look at:

- [MIT introduction to deep learning](http://introtodeeplearning.com/)
- [Nvidia resources on deep learning](https://www.nvidia.com/en-us/deep-learning-ai/), and [developer site](https://developer.nvidia.com/)
- the courses taught at [Master Datascience Paris Saclay](http://datascience-x-master-paris-saclay.fr/) and available on https://github.com/m2dsupsdlclass/lectures-labs

----------------------------------
## 1. Preliminary installation ##

### Step 1.a Connect to a cluster node

To configure the environment, we only need one core of one node.

Request such resource from the iris-cluster:
```bash
[]$ ssh iris-cluster
[]$ si -n1 -c1
[]$ # Inspect your resources, several ways:
[]$ scontrol show job $SLURM_JOB_ID
[]$ sj $SLURM_JOB_ID  # alias for the above
[]$ env | grep SLURM  # similar information via env variables
```

### Step 1.b Prepare python virtualenv(s)

Our working environment will consist of:

- Python 3,
- CUDA toolkit for the GPU tests,
- Tensorflow, both for CPU and GPU.

Tensorflow comes in different versions for CPU and GPU.
We install both, and will select the correct one (matching the reserved node) before running the code.

To do so, we can rely on virtualenv.
If you have never used virtualenv before, please have a look at [Python1 tutorial](http://ulhpc-tutorials.readthedocs.io/en/latest/python/basics/).

```bash
[]$ # Load the required version of python, and the CUDA libraries:
[]$ module load lang/Python/3.6.4-foss-2018a system/CUDA numlib/cuDNN
[]$ module list # ml
```
To quickly recover our configuration, when developing interactively, we can save a module configuration:
```bash
[]$ # Save the module in a specific module list named 'tf':
[]$ module save tf
[]$ module purge
[]$ module restore tf
```
Set up virtualenv:
```bash
[]$ # Create a common directory for your python 3 environments:
[]$ mkdir ~/venv && cd ~/venv
[]$ # for the CPU tensorflow version
[]$ virtualenv tfcpu
[]$ # for the GPU version
[]$ virtualenv tfgpu
```

### Step 1.c. Install Tensorflow

See also [installation notes](https://www.tensorflow.org/install/)

You should have 2 virtualenvs created: tfcpu and tfgpu.

First install for the *CPU environment* (that matches the current reservation request):

```bash
[]$ source ~/venv/tfcpu/bin/activate
(tfcpu) []$ pip install tensorflow  # check the prompt for the environment
(tfcpu) []$ # Check the installation:
(tfcpu) []$ python -c "import tensorflow as tf; print('tf:{}, keras:{}.'.format(tf.__version__, tf.keras.__version__))"
```

At the time of writing, recent versions of numpy broke a tensorflow tutorial source file.
In case of exception when unpickling the training data, apply a patch [`imdb.patch`](./imdb.patch), listed below:
```
--- lib/python3.6/site-packages/tensorflow/python/keras/datasets/imdb_.py	2019-05-16 14:51:36.074289000 +0200
+++ lib/python3.6/site-packages/tensorflow/python/keras/datasets/imdb.py	2019-05-16 14:52:12.984972000 +0200
@@ -82,7 +82,7 @@
       path,
       origin=origin_folder + 'imdb.npz',
       file_hash='599dadb1135973df5b59232a0e9a887c')
-  with np.load(path) as f:
+  with np.load(path, allow_pickle=True) as f:
     x_train, labels_train = f['x_train'], f['y_train']
     x_test, labels_test = f['x_test'], f['y_test']
```
Apply the patch:
```bash
(tfcpu) []$ cp imdb.patch $VIRTUAL_ENV
(tfcpu) []$ cd $VIRTUAL_ENV
(tfcpu) []$ patch -p0 <imdb.patch
```

Then install for the *GPU environment*.
You can install from a CPU node, but not execute any GPU specific code.
To change virtual environments, you do not need to `deactivate` the tfcpu environment first:

```bash
[]$ source ~/venv/tfgpu/bin/activate
(tfgpu) []$ pip install tensorflow-gpu  # check the prompt for the environment
(tfgpu) []$ # Check the installation:
(tfgpu) []$ python -c "import tensorflow as tf; print('tf:{}, keras:{}.'.format(tf.__version__, tf.keras.__version__))"
```
Apply the patch again (patch file should still be there):
```bash
(tfgpu) []$ cd $VIRTUAL_ENV
(tfgpu) []$ patch -p0 <imdb.patch
```

-----------------------------------------------------------------
## 2. Interactive development of a Tensorflow application ##

In this step, we will develop a tensorflow model on a CPU node (more available), and execute it interactively on both CPU and GPU nodes.

### Pre-requisites

Connect to an iris cluster node, GPU is not required.
See above to request such a resource.

If Python is not already loaded:
```bash
[]$ module restore tf
```
If your virtual environment is not available, activate it:
```bash
[]$ source ~/venv/tfcpu/bin/activate
(tfcpu) []$
```

### Tensorflow example model

Head to the
[TensorFlow text classification tutorial](https://www.tensorflow.org/tutorials/keras/basic_text_classification)
and follow the steps there to assemble a Tensorflow application.

You should end up with a standalone python program that defines, trains and predicts a model.

### GPU interactive execution

Here, we will execute the functioning program developed above on a GPU node, interactively.

Request such resource from the iris-cluster:
```bash
[]$ ssh iris-cluster
[]$ si-gpu -G 1 -n1 -c1
```
Load the modules, and the choose the proper python environment:
```bash
[]$ module restore tf
[]$ source ~/venv/tfgpu/bin/activate
(tfgpu) []$
```
Run the *same* tensorflow tutorial program that was developed under the CPU node.

-----------------------------------------------------------------
## 3. Batch execution of a Tensorlow application ##

Now, we will execute the same Tensorflow program in batch mode, on a CPU or GPU node.

One program from the tutorial is here [imdb-train.py](./imdb-train.py).

To run in batch mode, you need a *launcher* file, that is passed as an argument to `sbatch`.
From the previous interactive sessions, you can write such launcher files.

If you get stuck, here is an initial version for the CPU node:
```bash
#!/bin/bash -l

#SBATCH --job-name="CPU imdb"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:10:00
#SBATCH --partition=batch
#SBATCH --qos=normal

module restore tf
source ~/venv/tfcpu/bin/activate
srun python imdb-train.py
```

Adjust the script for the GPU node.
(See [tfgpu.sh](./tfgpu.sh) for one example)

What happens if you change the resources assigned to this job? (number of cores, number of GPUs)
