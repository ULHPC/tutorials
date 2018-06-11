[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/python/basics/tutorial_python.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/python/basics/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/python/basics) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# UL HPC Tutorial: Prototyping with Python

      Copyright (c) 2017-2018 UL HPC Team <hpc-sysadmins@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/python/basics/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/python/basics/tutorial_python.pdf)

Python is a high-level interpreted language widely used in research. It lets you work quickly and comes with a lot of available packages which give more useful functionalities.

In this tutorial, we are going to explain the steps to run a Python script on the cluster and install a Python package as a user. We will also create a virtual environment and switch from one to the other. We will show how to use different versions of Python on a node. Finally, we will parallelize your code with Scoop and compile it in C to fasten its execution.

## Examples used in this tutorial

### Example 1

The first example used in this tutorial is fully inspired from [PythonCExtensions](https://github.com/mattfowler/PythonCExtensions). This code simply computes the mean value of an array of random numbers. The naïve code used to compute the mean of an array is:

```python
def mean(lst):
    return sum(lst) / len(lst)


def standard_deviation(lst):
    m = mean(lst)
    variance = sum([(value - m) ** 2 for value in lst])
    return math.sqrt(variance / len(lst))
```

The variable will be the size of the array on which we want to compute the mean. The idea is to reduce the time used to compute this value by using libraries (**numpy**) or compile the code in C.

### Example 2

The second example used in this tutorial comes from [Scoop example computation of pi](http://scoop.readthedocs.io/en/0.7/examples.html#computation-of). We will use a [Monte-Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method) to compute the value of pi. As written in the Scoop documentation, it spawns two pseudo-random numbers that are fed to the hypot function which calculates the hypotenuse of its parameters. This step computes the Pythagorean equation (\sqrt{x^2+y^2}) of the given parameters to find the distance from the origin (0,0) to the randomly placed point (which X and Y values were generated from the two pseudo-random values). Then, the result is compared to one to evaluate if this point is inside or outside the unit disk. If it is inside (have a distance from the origin lesser than one), a value of one is produced (red dots in the figure), otherwise the value is zero (blue dots in the figure). The experiment is repeated *tries* number of times with new random values.

![alt text](http://scoop.readthedocs.io/en/0.7/_images/monteCarloPiExample.gif "Monte-Carlo pi calculation example")

The variable here will be the number of workers (cores on which the script runs) compared to the time of execution.

## Python usage

In this part we will simply run our Python script on the UL HPC platform, on a single node.

### Get all the scripts

Clone the UL HPC python tutorial under your home directory on the **Iris** cluster. If you have cloned it before, simply run `git pull` to update it to the latest version.

```
(laptop)$> ssh iris-cluster
(access)$> git clone https://github.com/ULHPC/tutorials.git
(access)$> cd tutorials/
(access)$> git stash && git pull -r && git stash pop
```

All the scripts used in this tutorial can be found under `tutorials/advanced/Python/`.

### Execute your first python script on the cluster (Example 1)

First, connect to `iris-cluster` and go to example 1:

```
(laptop)$> ssh iris-cluster
(access)$> cd tutorials/advanced/Python/example1/
```

To run your script **interactively** on the cluster, you should do:

```
(access)>$ si
(iris-001)$> python example1.py
```

You should see the output of your script directly written in your terminal. It prints the length of the array and the number of seconds it took to compute the standard deviation 10,000 times.

To run your script in a **passive** way, you should create a batch script to run your python script.

* Create a `example1.sh` file under `tutorials/advanced/Python/example1/`.
* Edit it by using your favorite editor (`vim`, `nano`, `emacs`...)
* Add a shebang at the beginning (`#!/bin/bash`)
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

**HINT:** You can find the answer under `tutorials/advanced/Python/example1/answer/example1.sh.answer`.

## Compare version of Python

You can switch between several version of Python that are already install on UL HPC iris cluster. To list the versions available, you should use this command on a compute node:

```
(iris-001)$> module spider Python
```

**QUESTIONS:**

* What are the versions of Python available on Iris cluster ? On Gaia cluster ?
* Which toolchains have been used to build them ?

Here we will compare the performance of Python 2.7 and Python 3.

Here are the steps to compare 2 codes:

* Go to `tutorials/advanced/Python/example2`
* Create a batch script named `example2.sh`
* Edit it with your favorite editor (`vim`, `nano`, `emacs`...)
* Add a shebang at the beginning (`#!/bin/bash`)
* Add **#SBATCH** parameters
  * `1` core
  * `example2` name
  * maximum `10m` walltime
  * logfile under `example2.out`
* Load Python version 2.7
* Execute the script a first time with this version of Python
* Load Python version 3
* Execute the script a second time with this Python version.
* Check the content of the file `example2.out` and identify the 2 executions and the module load.

**QUESTIONS**

* What is the fastest version of Python ?
* There are both `foss` and `intel` compiled versions of Python available on the Iris cluster. Modify your script to compare their execution time. Which is the fastest one ?

**HINT**

* You can use `module load` command to load a specific version of Python.
* An example of a BATCH script can be found under `tutorials/advanced/Python/example2/answer/example2.sh.answer`

## Use a library to optimize your code

In this part we will try to use [Numpy](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html), a Python library, to optimize our code.

In `tutorials/advanced/Python/example3/example3.py` you should see a version of the previous script using Numpy.

Try to execute the script on iris cluster in **interactive** mode.

```
(access)$> si
(iris-001)$> python example3.py
```

**QUESTIONS**

* Why did the execution fail ? What is the problem ?

We need to install the numpy library. Hopefully, numpy is available as a module on iris cluster. Use the commands from `example2` to:

* Load `numpy` module
* Execute your script on the node

```
(access)$> si
(iris-001)$> module spider numpy
(iris-001)$> module load math/numpy
(iris-001)$> python example3.py
```

As you can see, there is only one version of numpy available with module. If we want to use a more recent or older version of numpy, we can install it ourselves in our home directory. For that we will use the `pip` tool.

`pip` is a package manager for Python. With this tool you can manage Python packages easily: install, uninstall, list, search packages or upgrade them. If you specify the `--user` parameter, the package will be installed under **your home directory** and will be available on all the compute nodes. Let's install numpy using `pip`.

```
(access)$> si
(iris-001)$> pip install --user numpy==1.13.0
(iris-001)$> python -c "import numpy as np; print np.__version__"
```

**QUESTIONS**

* Which execution is faster between *numpy* code (example3.py) and *naïve* code (example1.py) ?
* Why do you think that numpy is not as powerful as intended ? Which parameter can we change to compare the performances ?

## Create virtual environment to switch between several versions of a package

Here comes a very specific case. Sometimes you have to use tools which depend on a specific version of a package. You probably don't want to uninstall and reinstall the package with `pip` each time you want to use one tool or the other.

Virtualenv allows you to create several environments which will contain their own list of Python packages. The basic usage is to **create one virtual environment per project**.

In this tutorial we will create a new virtual environment for the previous code in order to install a different version of numpy and check the performances of our code with it.

First of all, install `virtualenv` package using pip:

```
(access)$> si
(iris-001)$> pip install --user virtualenv
```

Now you can create your environment for this project. Name it `numpy12`.

```
(iris-001)$> cd ~/tutorials/advanced/Python/example3/
(iris-001)$> virtualenv numpy12
```

So now you should be able to active this environment with this `source` command. Please notice the `(numpy12)` present in your prompt that indicates that the `numpy12` environment is active.

```
(iris-001)$> source numpy12/bin/activate
(numpy12)(iris-001)$>
```

**QUESTIONS**

* Using `pip freeze`, what are the modules available before the activation of your virtual environment ?
* What are the module available after ?
* What version of python is used inside the virtual environment ? Where is it located ? (You can use `which` command.)

To exit a virtual environment run the `deactivate` command.

So now, we can install a different numpy version inside our virtual environment. Check that the version installed corresponds to numpy 1.12.

```
(iris-001)$> source numpy12/bin/activate
(numpy12)(iris-001)$> pip install numpy==1.12
(numpy12)(iris-001)$> python -c "import numpy as np; print np.__version__"
(numpy12)(iris-001)$> deactivate
(iris-001) python -c "import numpy as np; print np.__version__"
```

Now you can adapt your script to load the right virtualenv and compare the performance of different versions of numpy.

**QUESTIONS**

* Check the size of numpy12 folder. Why is it so big ? What does it contain ?

## Compile your code in C language

C language is known to be very powerful and to execute faster. It has to be compiled (typically using GCC compiler) to be executed. There exist many tools that can convert your Python code to C code to benefit from its performances (**Cython**, **Pythran**, ...).

The goal of this part is to adapt our naïve code and use the **Pythran** tool to convert it to C code. This code will then be imported as a standard Python module and executed.

The code can be found under `tutorials/advanced/Python/example4/example4.py`.

* Open the `example4.py` file
* Referring to [Pythran documentation](https://github.com/serge-sans-paille/pythran), add a comment before the `standard_deviation` function to help pythran to convert your python function into a C one.
  * Parameter should be a list of float
  * Function name should be `standard_dev`

```
#code to insert in example4.py

#pythran export standard_dev(float list)
def standard_dev(lst):
```

* Be sure to have `pythran` installed! If not, use `pip install --user pythran` command (within a job) to install it in your home directory.
* Compile your code using pythran:

```
(iris-001)$> pythran example4.py -o std.so # NEVER COMPILE ON ACCESS
(iris-001)$> python -c "import std" # this imports the newly generated module with C implementation
```

* Have a look at `c_compare.py` that contains the code to
  * import your module
  * and execute the mean function from this module on a random array
* Execute your code on a node and compare the execution time to the other one.

**QUESTIONS**

* What is the fastest execution ? Why ?
* Where can I find the code that has been generated from my Python script ?

**HINT:** If you run `pythran example4.py -e -o std.c` it will generate the C code. Have a look at the `*.c` files in your directory.

### Overview graph of runtimes

![alt-text](https://github.com/ULHPC/tutorials/raw/devel/advanced/Python/time_vs_array_size.jpg)

## Use Scoop to parallelize execution of your Python code with Slurm

In this part, we will use Scoop library to parallelize our Python code and execute it on iris cluster. This example uses the Monte-Carlo algorithm to compute the value of pi. Please have a look at the top of this page to check how it works.

**Attention:** Scoop only works with the system python and neither with modules nor virtual environments. Make sure you start with a clean environment by starting a new job.

We will first have to install the scoop library using `pip`:

```
(access)$> si
(iris-001)$> pip install --user filelock
(iris-001)$> pip install --user https://github.com/soravux/scoop/archive/master.zip
```

Scoop comes with direct Slurm bindings. If you run your code on a single node, it will try to use the most cores that it can. If you have reserved several nodes, it will use all the nodes of your reservation and distribute work on it.

You can specify the number of cores to use with the `-n` option in scoop.

We will write a batch script to execute our python script. We want to compare time of execution to the number of workers used in scoop. We want to go from 1 worker (single core) to 50 workers, increasing the worker number 1 by 1. As you can see, our script takes 1 parameter `x` in input which corresponds to the number of workers.

The batch script should contain:

* 50 tasks reserved
* maximum execution time of 35m
* name of the job should be `scoop`
* a job array which goes from 1 to 56 (maximal number of core on 2 nodes)
* a minimum number of 2 nodes reserved
* a call to `python -m scoop [...]` to call the script with increasing number of cores reserved (`$SLURM_ARRAY_TASK_ID`)
* a command to disable concurrent run of this job
* be the only user to use those ressources to avoid conflicting with other scoop users (see `--exclusive` option of sbatch)
* output file should go to `scoop.log`

**HINT** Have a look at `tutorials/advanced/Python/example5/scoop.sh` for the batch script example

Run this script with `sbatch` command. Check the content of `scoop.log` using `tail scoop.log` to see if everything is going well.

When your job is over, you can use `make graph` command to generate the graph.

**QUESTIONS**

* What is the correlation between number of workers and execution time ?
* Use what you learned in the previous part to optimize your code!
