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
## Pre-requisites ##

Ensure you are able to [connect to the UL HPC clusters](https://hpc.uni.lu/users/docs/access.html)
**For all tests and compilation, you MUST work on a computing node**

You'll need to prepare the data sources required by this tutorial once connected

``` bash
### ONLY if not yet done: setup the tutorials repo
# See http://ulhpc-tutorials.rtfd.io/en/latest/setup/install/
$> mkdir -p ~/git/github.com/ULHPC
$> cd ~/git/github.com/ULHPC
$> git clone https://github.com/ULHPC/tutorials.git
$> cd tutorials
$> make setup          # Initiate git submodules etc...
```

Now you can prepare a dedicated directory to work on this tutorial:

```bash
$> cd ~/git/github.com/ULHPC/tutorials/deep_learning
```

**Advanced users only**: rely on `screen` (see  [tutorial](http://support.suso.com/supki/Screen_tutorial) or the [UL HPC tutorial](https://hpc.uni.lu/users/docs/ScreenSessions.html) on the  frontend prior to running any `oarsub` or `srun/sbatch` command to be more resilient to disconnection.

Finally, be aware that the latest version of this tutorial is available on
[Github](https://github.com/ULHPC/tutorials/tree/devel/deep_learning/) and on

<http://ulhpc-tutorials.readthedocs.io/en/latest/deep_learning/>

This hands-on is **not** about learning Machine Machine Learning (ML)/Deep Learning (DL).
For that, we encourage you to take a look at the **EXCELLENT** courses being taught at as part of [Master Datascience Paris Saclay](http://datascience-x-master-paris-saclay.fr/) and available on

https://github.com/m2dsupsdlclass/lectures-labs

In particular, start with the
[Introduction to Deep Learning](https://m2dsupsdlclass.github.io/lectures-labs/slides/01_intro_to_deep_learning/index.html#1) _by_ Charles Ollion and Olivier Grisel.

Just as a short reminder, here are the main steps in ML:

* __Step 0 - 1__: Asking the right questions
* __Step 2 - 4__: Getting the right data
* __Step 5 - 7__: Finding patterns in the data
* __Step 8 - 9__: Checking the patterns work on new data
* __Step 10__:    Building a production-ready system
* __Step 11__:    Making sure that launching is a good idea
* __Step 12__:    Keeping a production ML system reliable over time

Math is finally just serving for

* Finding pattern (within old data)
* Assessing model performance (over the new data)

Finally, the **Biggest pitfalls** you should try to avoid are:

* _overfitting_: harder to detect but the most nightmare for ML (corresponds to having extracted a model out of noise)
* _underfitting_

----------------------------------
## 1. Preliminary installations ##

This tutorial relies on a [Jupiter notebook](https://jupyter.org/).

Due to an issue on iris, you will need a specific script to reverse a node with X11 forwarding. You can clone this script [on jbornschein srunx11 repo on Github](https://github.com/jbornschein/srun.x11.git).

```bash
# Connect to iris using the X11 forwarding AND a local SOCKS5 proxy on port 1080
$> ssh -D 1080 -Y iris-cluster
$> git clone https://github.com/jbornschein/srun.x11.git
# Reserve an node interactively
$> ./srun.x11/srun.x11 -p interactive --exclusive -c 28
```

Then we will work in a dedicated directory

``` bash
$> mkdir ~/tutorials/ML-DL
$> cd ~/tutorials/ML-DL
$> ln -s ~/git/hub.com/ULHPC/tutorials/deep_learning ref.d
```

As mentioned above, we are going to reuse some of the **EXCELLENT** labs materials taught at as part of [Master Datascience Paris Saclay](http://datascience-x-master-paris-saclay.fr/).

``` bash
$> pwd
$HOME/tutorials/ML-DL
$> git clone https://github.com/m2dsupsdlclass/lectures-labs/
```

### Step 1.a Prepare python virtualenv

If you have never used virtualenv before, please have a look at [Python1 tutorial](http://ulhpc-tutorials.readthedocs.io/en/latest/python/basics/).

```bash
# Load your prefered version of Python
$> module load lang/Python/2.7.13-foss-2017a
# Create a new virtualenv
$> pip install --user virtualenv
$> PATH=$PATH:$HOME/.local/bin
$> virtualenv dlenv
$> source dlenv/bin/activate
$> pip install -U pip   # Upgrade local pip
```
## Step 1.b. Install Tensorflow

See also [installation notes](https://www.tensorflow.org/install/)

Assuming you work within a `dlenv` virtualenv environment:

```
# ENSURE you are in dlenv environment
$> pip install --upgrade tensorflow
```

## Step 1.c. Install Jupyter

See <https://jupyter.org/install.html>

```bash
# install jupyter himself
$> pip install jupyter
# To use our virtualenv in the notebook, we need to install this module
$> pip install ipykernel
```

## Step 1.d. Configure Jupiter Kernel

In order to access to all the modules we have installed in the `dlenv` environment, we will need to create a new Kernel and use it inside Jupyter.

To do so, let's use `ipykernel`.

```bash
# Now run the kernel "self-install" script to create a new kernel nesusWS
$> python -m ipykernel install --user --name=dlenv
```

## Step 1.e. Launch jupyter notobook

You can now start the notebook.
To have access to it from the outside, we will need to run it on the correct IP of the node.

`/!\ IMPORTANT`: Please ensure that you are running the below commands inside the correct directory!

``` bash
$> cd ref.d/tensorflow/
```

This below commands can be now run.

* The `--ip` command permits to start the notebook with the correct IP.
* The `--no-browser` command is used to disable the openning of the browser after the start of the notebook.
* We use `--generate-config` at first to generate a default configuration.
   - The default configuration is stored in `~/.jupyter/jupyter_notebook_config.py`

To make things easier, we will protect our Notebook with a password. You have just to choose a password after typing the `jupyter notebook password` command. A hash of your password will be stored in the jupyter config file.


``` bash
$> cd ref.d/tensorflow/
$> jupyter notebook --generate-config
$> jupyter notebook password
$> jupyter notebook --ip $(ip addr show em1 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1) --no-browser
[I 13:50:56.251 NotebookApp] Writing notebook server cookie secret to /run/user/5000/jupyter/notebook_cookie_secret
[I 13:50:56.478 NotebookApp] Serving notebooks from local directory: /mnt/irisgpfs/users/svarrette/git/hub.com/ULHPC/tutorials/deep_learning/tensorflow
[I 13:50:56.479 NotebookApp] 0 active kernels
[I 13:50:56.479 NotebookApp] The Jupyter Notebook is running at:
[I 13:50:56.479 NotebookApp] http://172.17.6.3:8888/
[I 13:50:56.479 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```

As done during the [Big Data tutorial on Spark](https://ulhpc-tutorials.readthedocs.io/en/latest/bigdata/#4-interactive-big-data-analytics-with-spark), we are going to use a SOCKS 5 Proxy Approach to access the browser to access the notbook (on `http://<IP>:8888`).
That means that:

* You should initiate an SSH connetion with `-D 1080` option to open on the local port 1080.
   - this was normally done at the very beginning of this talk, otherwise you can just restart one SSH session from another terminal.

```
(laptop)$> ssh -D 1080 -C iris-cluster
```

Now, install for example the [Foxy Proxy](https://getfoxyproxy.org/order/?src=FoxyProxyForFirefox)
extension for Firefox and configure it to use your SOCKS proxy:

* Right click on the fox icon
* Options
* **Add a new proxy** button
* Name: `ULHPC proxy`
* Informations > **Manual configuration**
  * Host IP: `127.0.0.1`
  * Port: `1080`
  * Check the **Proxy SOCKS** Option
* Click on **OK**
* Close
* Open a new tab
* Right click on the Fox
* Choose the **ULHPC proxy**

Now you should be able to access the Spark master website, by entering the URL `http://172.17.XX.YY:8888/` (adapt the IP -- in the above example, that would be `http://172.17.6.3:8888` -- just copy/paste what is provided to you in the shell.

When you have finished, don't forget to close your tunnel and disable FoxyProxy
on your browser.

__Alternatively__, if you're relunctant to install a plugin, need to create an SSH tunneling in order to access this URL on your laptop. That means that we will forward every local requests made on your localhost address of your laptop the cluster. The simpliest way to do so is to type this command if your are on Linux:

```bash
# ONLY if you don't want to use the SOCKS5 proxy approach
$> ssh -NL 8888:<IP OF YOUR NODE>:8888 iris-cluster
# In the above example, the command would be: ssh -NL 8888:172.17.6.3:8888 iris-cluster
```

Then, by accessing to your local port 8888 on the localhost address 127.0.0.1, you should see the Python notebook.


-----------------------------------------------------------------
## 2. MNIST Machine Learning (ML) and Deep ML using Tensorflow ##

You are ready to open the jupiter notebook.

In the advent where you want to restart the notebook after having lost en finished the connection, here are the instructions you should follow:

``` bash
$> ssh -Y -D 1080 iris-cluster
$> ./srun.x11/srun.x11 -p interactive --exclusive -c 28
$> cd ~/tutorials/ML-DL/
$> module load lang/Python/2.7.13-foss-2017a
$> source dlenv/bin/activate
$> jupyter notebook password
$> jupyter notebook --ip $(ip addr show em1 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1) --no-browser
```

Now you can access the notebook in your browser -- enter the set password.


### Overview

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

### Simple MNIST classifier.

Open now the `ref.d/tensorflow/mnist-1-simple.ipynb`

**`/!\ IMPORTANT`** You will need to **change** the kernel to `dlenv`.  Enter it when the message "Kernel not found" appears. You can also use the 'Kernel -> Change kernel' menu


### Deep MNIST classifier using convolutional layer

You can now open `ref.d/tensorflow/mnist-2-deep_convolutional_NN.ipynb`

**`/!\ IMPORTANT`** You will need to **change** the kernel to `dlenv`.  Enter it when the message "Kernel not found" appears. You can also use the 'Kernel -> Change kernel' menu

--------------------------------------------
## 3. Training Neural Networks with Keras ##

Let's first install `keras` -- you will need to open another tab session with `screen` (using `CTRL-A c` (for create))

``` bash
$> module load lang/Python/2.7.13-foss-2017a
$> source dlenv/bin/activate
$> pip install keras
```

You can now open in the jupyter `lectures-labs/labs/01_keras/Intro\ Keras.ipynb`


----------------------------------------------
## 4. Pretrained Models for Computer Vision ##

You can now open in the jupyter `lectures-labs/labs/01_keras/Intro\ pretrained\ models\ for\ computer\ vision.ipynb`
