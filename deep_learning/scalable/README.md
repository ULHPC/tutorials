[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/deep_learning/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/deep_learning/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/deep_learning/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)


# Scalable Deep Learning on the UL HPC Platform

     Copyright (c) 2013-2019 UL HPC Team  <hpc-sysadmins@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/deep_learning/basics/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/deep_learning/basics/slides.pdf)

The objective of this tutorial is to practice running Horovod (and Keras/TensorFlow) on the UL HPC [iris cluster](https://hpc.uni.lu/systems/iris/).

It's important that you read the [slides](https://github.com/ULHPC/tutorials/blob/devel/deep_learning/scalable/slides.pdf?raw=true) first.

## Horovod with TensorFlow, multi-node & multi-GPU tests

* As an initial test, you will now use the following launcher to:
  - reserve 2 GPU nodes and all their GPUs (8) - edit the launcher to match this
  - start Horovod through its `horovodrun` wrapper

```bash
#!/bin/bash -l
#SBATCH -J HorovodTFGPU
#SBATCH -o %x_%j.out
#SBATCH -N 2
#SBATCH -n 4
#SBATCH --gres=gpu:4
#SBATCH -t 1:0:0
#SBATCH -p gpu

## The following only needed during HPC School 2019.06
module load swenv/default-env/devel

## Load TF and Horovod that link to CUDA, cuDNN & NCCL
module load lib/TensorFlow/1.13.1-fosscuda-2019a-Python-3.7.2
module load tools/Horovod/0.16.3-fosscuda-2019a-Python-3.7.2

## Create tests directory and clone the TF benchmarks inside
mkdir $SCRATCH/tests-horovod && cd $SCRATCH/tests-horovod
git clone https://github.com/tensorflow/benchmarks

## Horovod execution
horovodrun -np $SLURM_NTASKS \
    python scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
      --model resnet101 --batch_size 64 --variable_update horovod
```

* Now check:
  - how many images/sec did the benchmark show at the end of its execution?
  - what results do you get from a single node (4 GPUs)? and from a single GPU?

* If you load the non-accelerated versions for TF and Horovod (the ones on the `foss` instead of `fosscuda` toolchain)
  - what result do you get from a regular compute node without GPUs (use the `batch` partition) when using it fully, i.e. 28 cores?
  - how many full regular nodes do you need to use to replicate the benchmark result from a single accelerated node with its 4 GPUs?


## Horovod with Keras and TensorFlow

For this part we will use the (excellent) [SC18 Tutorial: Deep Learning At Scale](https://github.com/NERSC/sc18-dl-tutorial).

You will need to `git clone https://github.com/NERSC/sc18-dl-tutorial` on the Iris cluster (preferrably under your $SCRATCH).
Then we will need to adapt its input configuration files under `configs` and the launcher `scripts`.

You will find under the current (UL HPC) [tutorial's repository ](https://github.com/ULHPC/tutorials/tree/devel/deep_learning/scalable) customized files to be used for the Iris cluster:

```bash
configs/
├── cifar10_cnn.yaml
├── cifar10_resnet.yaml
├── hello.yaml
└── imagenet_resnet.yaml
scripts/
├── cifar_cnn.sh
├── cifar_resnet.sh
├── imagenet_resnet.sh
└── setup.sh
```

Typically you will start by launching the `cifar-cnn.sh` example, and will quickly discover it's running slow (check the appropriate output in `logs/`.
What will you need to adapt? What's different from the `*_resnet.sh` launchers? (take a look at what `train.py` does in `--distributed` mode).
