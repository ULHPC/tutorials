[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/python/basics/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/python/basics/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/python/basics) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# UL HPC Tutorial: Python basics

      Copyright (c) 2018-2021 UL HPC Team <hpc-sysadmins@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/python/basics/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/python/basics/slides.pdf)

Python is a high-level interpreted language widely used in research. It lets you work quickly and comes with a lot of available packages which give more useful functionalities.

In this tutorial, we are going to explain the steps to run a Python script on the cluster and install a Python package as a user. We will also create a virtual environment and switch from one to the other. We will show how to use different versions of Python on a node. We will speed up the code using packages and by compiling it in C. Finally, we will install an independent Python version using the conda package manager.

## Overview

### Requirements

* Access to the UL HPC clusters.
* Basic knowledge of the linux command-line.
* Basic programming knowledge.
* Basic Python knowledge.
* Running passive SLURM jobs using launcher scripts.

### Questions

* How can I run Python scripts on the cluster?
* What Python versions are available on the cluster and how can I use them?
* How can I speed up my python code?
* How can I install python packages?
* How can I manage different versions of Python or packages?

### Objectives

* Run Python scripts on the cluster.
* See the difference between Python versions.
* Speed up code using packages.
* Speed up code by compiling it in C.
* Install Python packages.
* Switch between different Python and package versions using a virtual environment.
* Create an independent Python installation with conda.

## Example: compute standard deviation

The first example used in this tutorial is fully inspired from [PythonCExtensions](https://github.com/mattfowler/PythonCExtensions). This code computes the standard deviation of an array of random numbers. The naïve code used to compute the standard deviation of an array (`lst`) is:

```python
def mean(lst):
    return sum(lst) / len(lst)


def standard_deviation(lst):
    m = mean(lst)
    variance = sum([(value - m) ** 2 for value in lst])
    return math.sqrt(variance / len(lst))
```

The variable will be the size of the array on which we want to compute the standard deviation. The idea is to reduce the time used to compute this value by using libraries (**numpy**) or compile the code in C.

## Python usage

In this part we will simply run our Python script on the UL HPC platform, on a single node.

### Get all the scripts

Clone the UL HPC python tutorial under your home directory on the **Iris** or **Aion** cluster. If you have cloned it before, simply run `git pull` to update it to the latest version.

```
(laptop)$> ssh aion-cluster
(access)$> git clone https://github.com/ULHPC/tutorials.git
(access)$> cd tutorials/
(access)$> git stash && git pull -r && git stash pop
```

All the scripts used in this tutorial can be found under `tutorials/python/basics`.

### Execute your first python script on the cluster (Example 1)

First, connect to `aion-cluster` and go to example 1:

```
(laptop)$> ssh aion-cluster
(access)$> cd tutorials/python/basics/example1/
```

To run your script **interactively** on the cluster, you should do:

```
(access)>$ si
(node)$> python example1.py
```

You should see the output of your script directly written in your terminal. It prints the length of the array and the number of seconds it took to compute the standard deviation 10,000 times.

To run your script in a **passive** way, you should create a batch script to run your python script.

* Create a `example1.sh` file under `tutorials/advanced/Python/example1/`.
* Edit it by using your favorite editor (`vim`, `nano`, `emacs`...)
* Add a shebang at the beginning (`#!/bin/bash -l`)
* Add **#SBATCH** parameters (see [Slurm documentation](https://hpc.uni.lu/users/docs/slurm.html))
  * `1` core
  * `example1` name
  * maximum `10m` walltime
  * logfile under `example1.out`

Now run the script using

```
(access)$> sbatch example1.sh
```

Now, check that the content of `example1.out` corresponds to the expected output (in interactive mode).

**HINT:** You can find the answer under `tutorials/python/basics/example1/answer/example1.sh.answer`.

## Different versions of Python

There are multiple versions of Python installed on the UL HPC clusters.

First, you have the Python provided with the operating system, e.g. the default `python`. Since you cannot be sure which version this is, you should check it with:

```
(node)$> python --version
```

Usually, you have both versions 2.7 and 3 available this way:

```
(node)$> python2 --version
(node)$> python3 --version
```

Additionally, we have newer versions of Python available through the modules. To list these versions, you should use this command on a compute node:

```
(node)$> module avail lang/Python
```

**QUESTIONS:**

* What are the versions of Python available on Iris cluster? On Aion cluster? To update Iris to the same versions as Aion, you can run `resif-load-swset-devel`.
* Which toolchains have been used to build them?

You can load a specific Python version provided through the modules with `module load`:

```
(node)$> module load lang/Python/3.8.6-GCCcore-10.2.0
```

You can pick any of these Python versions and try to rerun `example1.py`.

For the rest of the tutorial we will use the Python 3 version from the modules.

**IMPORTANT**:

* Python code is not necessarily compatible between versions 2 and 3.
* For many packages recent versions are only available for Python 3.
* Make sure to always use the same Python version (and package versions) when running your code or workflow.

## Use a package to optimize your code

In this part we will try to use [Numpy](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html), a Python package, to optimize our code.

In `tutorials/python/basics/example3/example3.py` you should see a version of the previous script using Numpy.

Try to execute the script on the Iris or Aion cluster in **interactive** mode.

```
(node)$> module purge
(node)$> module load lang/Python/3.8.6-GCCcore-10.2.0
(node)$> python example3.py
```

**QUESTIONS**

* Why did the execution fail ? What is the problem ?

We need to install the numpy library. We can install it ourselves in our home directory. For that we will use the `pip` tool.

`pip` is a package manager for Python. With this tool you can manage Python packages easily: install, uninstall, list, search packages or upgrade them. If you specify the `--user` parameter, the package will be installed under **your home directory** and will be available on all the compute nodes. You should also use `--no-cache` to prevent pip from searching in the cache directory which can be wrongly populated if you deal with several version of Python. Let's install numpy using `pip`.

```
(node)$> python -m pip install --no-cache --user numpy==1.16
(node)$> python -m pip show numpy
(node)$> python -m pip install --no-cache --user numpy==1.21
(node)$> python -m pip show numpy
```

Notice that with pip you can only have one version of numpy installed at a time. In the next section, we will see how to easily switch between several versions of numpy by using **vitualenv**.

You can now run **example3.py** code and check its execution time.

**QUESTIONS**

* Which execution is faster between *numpy* code (example3.py) and *naïve* code (example1.py)?
* Why do you think that numpy is not as powerful as intended? Which parameter can we change to compare the performances?

**NOTES**

* Numpy is also available from the `lang/SciPy-bundle` modules, tied to different Python versions. Check `module list` to see which Python version was loaded along the SciPy bundle.

## Create virtual environment to switch between several versions of a package

Here comes a very specific case. Sometimes you have to use tools which depends on a specific version of a package. You probably don't want to uninstall and reinstall the package with `pip` each time you want to use one tool or the other.

**Virtualenv** allows you to create several environments which will contain their own list of Python packages. The basic usage is to **create one virtual environment per project**.

In this tutorial we will create a new virtual environment for the previous code in order to install a different version of numpy and check the performances of our code with it.

Create two virtual environments for your project. They will contain two different versions of numpy (1.21 and 1.16). Name then`numpy21` and `numpy16`, respectively.

```
(node)$> cd ~/tutorials/python/basics/example3/
(node)$> python3 -m venv numpy21
(node)$> python3 -m venv numpy16
```

So now you should be able to active any of these environments with this `source` command. Please notice the `(numpy21)` present in your prompt that indicates that the `numpy21` environment is active. You can use `deactivate` command to exit the virtual environment.

```
(node)$> source numpy21/bin/activate
(numpy21)(node)$> # You are now inside numpy21 virtual environment
(numpy21)(node)$> deactivate
(node)$> source numpy16/bin/activate
(numpy16)(node)$> # You are now inside numpy16 virtual environment
```

**QUESTIONS**

* Using `python -m pip freeze`, what are the modules available before the activation of your virtual environment?
* What are the module available after?
* What version of python is used inside the virtual environment ? Where is it located ? (You can use `which` command.)

To exit a virtual environment run the `deactivate` command.

So now, we can install a different numpy version inside each of your virtual environments. Check that the version installed corresponds to numpy 1.21 for *numpy21* and numpy 1.16 in *numpy16*.

```
# Go inside numpy21 environment and install numpy 1.21
(node)$> source numpy21/bin/activate
(numpy21)(node)$> python -m pip install numpy==1.21
(numpy21)(node)$> python -m pip show numpy
(numpy21)(node)$> deactivate

# Go inside numpy16 environment and install numpy 1.16
(node)$> source numpy16/bin/activate
(numpy16)(node)$> python -m pip install numpy==1.16
(numpy16)(node)$> python -m pip show numpy
(numpy16)(node)$> deactivate
```

Now you can write a batch script to load the right virtualenv and compare the performance of different versions of numpy.

Here are the steps to compare the two versions:

* Go to `tutorials/python/basics/example3`
* Create a batch script named `numpy_compare.sh`
* Edit it with your favorite editor (`vim`, `nano`, `emacs`...)
* Add a shebang at the beginning (`#!/bin/bash -l`)
* Add `#SBATCH` parameters
  * `1` core
  * `numpy_compare` name
  * maximum `10m` walltime
  * logfile under `numpy_compare.out`
* Activate *numpy16* environment.
* Execute `numpy_compare.py` a first time with this version of numpy.
* Deactivate environment
* Activate *numpy21* environment..
* Execute the script a second time with this numpy version.
* Check the content of the file `numpy_compare.out` and identify the two executions.

**QUESTIONS**

* Check the size of numpy21 folder. Why is it so big ? What does it contain ?

## Compile your code in C language

C language is known to be very powerful and to execute faster. It has to be compiled (typically using GCC compiler) to be executed. There exist many tools that can convert your Python code to C code to benefit from its performances (**Cython**, **Pythran**, ...).

The goal of this part is to adapt our naïve code and use the **Pythran** tool to convert it to C code. This code will then be imported as a standard Python module and executed.

The code can be found under `tutorials/python/basics/example4/example4.py`.

* Open the `example4.py` file
* Referring to [Pythran documentation](https://github.com/serge-sans-paille/pythran), add a comment before the `standard_deviation` function to help pythran to convert your python function into a C one.
  * Parameter should be a list of float
  * Function name should be `standard_dev`

```
#code to insert in example4.py

#pythran export standard_dev(float list)
def standard_dev(lst):
```

* Create a new virtual environment, activate it and install `pythran`.
* Compile your code using pythran:

```
(node)$> pythran example4.py -e -o std.cpp # NEVER COMPILE ON ACCESS (only translate)
(node)$> pythran example4.py -o std.so # NEVER COMPILE ON ACCESS (compile)
(node)$> python -c "import std" # this imports the newly generated module with C implementation
```

* Have a look at `c_compare.py` that contains the code to
  * import your module
  * and execute the mean function from this module on a random array
* Execute your code on a node and compare the execution time to the other one.

**QUESTIONS**

* What is the fastest execution? Why?
* Where can I find the code that has been generated from my Python script?

**HINT:** If you run `pythran example4.py -e -o std.cpp` it will generate the C code. Have a look at the `*.cpp` files in your directory.

### Overview graph of runtimes

![alt-text](https://github.com/ULHPC/tutorials/raw/feature/2021_python-basics/python/basics/example4/answer/time_vs_array_size.jpeg)



## Install your own Python and create reproducible software environments with Conda

In this part we will use the [`conda` package manager](https://www.anaconda.com/) to install Python and the required packages.

> Conda is an open source package management system and environment management system that runs on Windows, macOS and Linux. Conda quickly installs, runs and updates packages and their dependencies. Conda easily creates, saves, loads and switches between environments on your local computer. It was created for Python programs, but it can package and distribute software for any language.
>
> Conda as a package manager helps you find and install packages. If you need a package that requires a different version of Python, you do not need to switch to a different environment manager, because conda is also an environment manager. With just a few commands, you can set up a totally separate environment to run that different version of Python, while continuing to run your usual version of Python in your normal environment.
>
> &mdash; <cite>[Conda  manual](https://docs.conda.io/en/latest/index.html)</cite>

It can encapsulate **software** and packages in environments, so you can have multiple different versions of a software installed at the same time and avoid incompatibilities between different tools. It also has functionality to easily port and replicate environments, which is important to ensure reproducibility of analyses.

You can think of it as an extension of Python virtualenv to all software, not just Python packages.

### Install conda on the cluster

Connect to the cluster and start an interactive job:

```bash
(laptop)$> ssh aion-cluster
(access)$> si
```

Create a backup of your `.bashrc` configuration file, since the conda installation will modify it:

```bash
(node)$> cp ~/.bashrc ~/.bashrc-$(date +%Y%m%d).bak
```

Install conda:

```bash
(node)$> wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
(node)$> chmod u+x Miniconda3-latest-Linux-x86_64.sh
(node)$> ./Miniconda3-latest-Linux-x86_64.sh
```

You need to specify your installation destination, e.g. `/home/users/sdiehl/tools/miniconda3`. You must use the **full** path and can**not** use `$HOME/tools/miniconda3`. Answer `yes` to initialize Miniconda3.

The installation will modify your `.bashrc` to make conda directly available after each login. To activate the changes now, run

```bash
(node)$> source ~/.bashrc
```

### Setup the environment

1. Update conda to the latest version:

   ```bash
   (node)$> conda update conda
   ```

2. Create a new empty conda environment and activate it:

   ```bash
   (node)$> conda create -n python_tutorial
   (node)$> conda activate python_tutorial
   ```

   After validation of the creation step and once activated, you can see that your prompt will now be prefixed with `(python_tutorial)` to show which environment is active.

3. Make sure Python does not pick up packages in your home directory:

   ```bash
   (python_tutorial)(node)$> export PYTHONNOUSERSITE=True
   ```
   Not applying this setting can cause erratic and unreproducible behaviour from conda, e.g. it will prefer outdated package versions in your home folder over newer ones in the active environment. If you are a regular (and exclusive) conda user, you might want to add this line to your `~/.bashrc` or `~/.bash_profile`.
4. Install Python and numpy:

   ```bash
   (python_tutorial)(node)$> conda install python numpy
   ```
   You can also just install Python with conda and then numpy with `pip`.

### Working with conda environments

You can list the packages installed in your current environment with:

```bash
(python_tutorial)(node)$> conda list
```

You can export your current environment to a yaml file with:

```bash
(python_tutorial)(node)$> conda env export > environment.yaml
(python_tutorial)(node)$> cat environment.yaml
```

This file can be shared or uploaded to a repository, to allow other people to recreate the same environment.

It contains three main items:

* `name` of the environment
* a list of `channels` (repositories) from which to install the packages
* a list of `dependencies`, the packages to install and optionally their version 

When creating this environment file via export, it will list the packages you installed and also all their dependencies and the dependencies of their dependencies down to the lowest level. However, when manually creating the file, it's sufficient to specify the top-level required packages or tools. All the dependencies will be installed automatically.

For our environment with Python and numpy, the most simple definition - if we do not care about versions - would be:

```yaml
name: python_tutorial
channels:
  - default
dependencies:
  - python
  - numpy
```

If you want to install numpy from `pip` instead, it would look like:

```yaml
name: python_tutorial
channels:
  - default
dependencies:
  - python
  - pip:
    - numpy
```

For reproducibility, it is advisable to always specify the versions, though.

```yaml
name: python_tutorial
channels:
  - default
dependencies:
  - python=3.9.7
  - numpy=1.21.2
```

Let us deactivate the environment, delete it and recreate it from the yaml file. You may use the exported yaml or create a minimal one like shown above and use this one.

```bash
(python_tutorial)(node)$> conda deactivate
(base)(node)$> conda remove --name python_tutorial --all
(base)(node)$> conda env create -f environment.yaml
(base)(node)$> conda activate python_tutorial
```

You can list available conda environments with:

```bash
(python_tutorial)(node)$> conda env list
```

### (Optional) Remove conda

If you want to stop conda from always being active:

```bash
(access)$> conda init --reverse
```

Alternatively, you can revert back to the backup of your `.bashrc` we created earlier. In case you want to get rid of conda completely, you can now also delete the directory where you installed it (default is `$HOME/miniconda3`).

## (Deprecated) Use Scoop to parallelize execution of your Python code with Slurm

In this part, we will use Scoop library to parallelize our Python code and execute it on iris cluster.

**WARNING**: Scoop uses `ssh` to spawn workers instead of `srun`, so no slurm steps are created. This also means that workers will neither see any loaded modules nor the virtual environment, if you use any.

The second example used in this tutorial comes from [Scoop example computation of pi](http://scoop.readthedocs.io/en/0.7/examples.html#computation-of). We will use a [Monte-Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method) to compute the value of pi. As written in the Scoop documentation, it spawns two pseudo-random numbers that are fed to the hypot function which calculates the hypotenuse of its parameters. This step computes the Pythagorean equation (\sqrt{x^2+y^2}) of the given parameters to find the distance from the origin (0,0) to the randomly placed point (which X and Y values were generated from the two pseudo-random values). Then, the result is compared to one to evaluate if this point is inside or outside the unit disk. If it is inside (have a distance from the origin lesser than one), a value of one is produced (red dots in the figure), otherwise the value is zero (blue dots in the figure). The experiment is repeated *tries* number of times with new random values.

![alt text](http://scoop.readthedocs.io/en/0.7/_images/monteCarloPiExample.gif "Monte-Carlo pi calculation example")

The variable here will be the number of workers (cores on which the script runs) compared to the time of execution.

**WARNING:** We will need to create a wrapper around SCOOP to manage the loading of modules and virtualenv before calling SCOOP module. It is a tricky part that will need some additional steps to be performed before running your script.

We will first have to install the scoop library using `pip`:

```
(access)$> si
(node)$> module load lang/Python/3.8.6-GCCcore-10.2.0
(node)$> python3 -m pip install --no-cache --user filelock
(node)$> python3 -m pip install --no-cache --user scoop
```

Scoop comes with direct Slurm bindings. If you run your code on a single node, it will try to use the most cores that it can. If you have reserved several nodes, it will use all the nodes of your reservation and distribute work on it.

You can specify the number of cores to use with the `-n` option in scoop.

We will write a batch script to execute our python script. We want to compare time of execution to the number of workers used in scoop. We want to go from 1 worker (`-n 1` for Scoop option) to 55 workers, increasing the worker number 1 by 1. As you can see, our script takes 1 parameter `x` in input which corresponds to the number of workers.

There will be 1 batch script. It should contain:

* 1 task per cpu
* maximum execution time of 35m
* name of the job should be `scoop`
* a variable `$NB_WORKERS=$1` for the number of workers to spawn, that takes the first command-line argument of the script as the values (use a value between 1 and 55 (maximal number of cores on 2 nodes on Iris is 56)
* will give `$NB_WORKERS` as an option to the scoop script.
* be the only user to use those resources to avoid conflicting with other scoop users (see `--exclusive` option of sbatch)
* (only on Iris) only execute the script on Skylake CPU nodes.

**HINT** Have a look at `tutorials/python/basics/example5/scoop_launcher.sh` for the batch script example

Run this script with `sbatch` command. Check the content of `scoop_*.log` to see if everything is going well. Also use `squeue -u $USER` to see the pending array jobs.

When your job is over, you can use `make graph` command to generate the graph.

**QUESTIONS**

* What is the correlation between number of workers and execution time ?
* Use what you learned in the previous part to optimize your code!

