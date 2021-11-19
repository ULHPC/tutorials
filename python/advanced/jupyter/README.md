[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/python/advanced/tutorial_python_advanced.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/python/advanced) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/python/advanced/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# UL HPC Tutorial: Use Jupyter notebook on UL HPC

      Copyright (c) 2021  UL HPC Team <hpc-team@uni.lu>

Authors: Clément Parisot (updated by Frédéric Pinel and Sébastien Varrette)


[![](https://github.com/ULHPC/tutorials/raw/devel/python/advanced/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/python/advanced/tutorial_python_advanced.pdf)

Python is a high-level interpreted language widely used in research. It lets you work quickly and comes with a lot of available packages which give more useful functionalities.

# Reserve a node

```bash
ssh [aion,iris]-cluster    # connect to the cluster
# Once on the clusters, ask for a interactive job (here for 1h)
si --time=01:00:00
# OR si-gpu --time=01:00:00 if a GPU is needed
```

# Load Python and install required modules in a virtualenv

If you have never used virtualenv before, please have a look at [Python1 tutorial](http://ulhpc-tutorials.readthedocs.io/en/latest/python/basics/).

```bash
# load your prefered **3.x** version of Python - 2.7 DEPRECACTED
module load lang/Python/3.8.6-GCCcore-10.2.0   # Load default python
# create virtual environment - distinguish between clusters
python -m venv venv_${ULHPC_CLUSTER}
# Note: You may want to centralize your virtualenvs under '~/venv/<name>_${ULHPC_CLUSTER}'
source venv_${ULHPC_CLUSTER}/bin/activate
```

Now we need to install all the modules needed. They are listed in the requirements.txt at the root of our tutorial directory. Here is the list of essentials ones:

```bash
# jupyter himself
pip install jupyter
# matplotlib to plot the graph inside your notebook
pip install matplotlib
# To use our virtualenv in the notebook, we need to install this module
pip install ipykernel
```

To save the installed packages:

```bash
pip freeze -l > requirements.txt
```

To install everything:

```bash
pip install -r requirements.txt
```

Now everything is installed properly.

# Create your own kernel and launch your Notebook

In order to access to all the modules we have installed inside your Notebook, we will need to create a new Kernel and use it inside Jupyter.

To do so, let's use `ipykernel`.

```bash
 python -m ipykernel install --user --name venv_${ULHPC_CLUSTER}
# The "venv" name here is to give your kernel a name and it will install it in your $HOME path. If a similarly named kernel exists, it will be overwritten.
# In case you would like your kernel to be installed into your active conda environment (<YOURCONDAPATH-PREFIX>/miniconda3/envs/jupyter_env/share/jupyter/kernels/), use the command below. This may be preferred as it encapsulates everything into a single environment, but would deviate from the virtualenv-based configuration above more than necessary for this tutorial.
#python -m ipykernel install --sys-prefix --name 'mylocalkernel'
# A completely custom specification of the path is *discouraged* as the resulting warning about the path not being in the "default" places and might hence not be found is very real. This means that the kernel can not be selected from the "New" dialog in the Jupyter interface. S. a. https://scipy-ipython.readthedocs.io/en/latest/install/kernel_install.html#kernels-for-different-environments for further information.
# Use only if you know what you do!!!
#python -m ipykernel install --prefix=./ --name 'myhyperlocalkernel'
```

Now everything is installed properly using conda.

Now we will have to start our first notebook. To have access to it from the outside, we will need to run it on the correct IP of the node. This simple command permits to start a new notebook with the correct IP. Please ensure that you are running the command inside the correct directory!

The `--no-browser` command is used to disable the openning of the browser after the start of the notebook. We use `--generate-config` at first to generate a default configuration. The default configuration is stored in `~/.jupyter/jupyter_notebook_config.py`

To make things easier, we will protect our Notebook with a password. You have just to choose a password after typing the `jupyter notebook password` command. A hash of your password will be stored in the jupyter config file.

```bash
#cd tutorials/python/advanced/jupyter # Only needed if you do not follow from above
jupyter notebook --generate-config
jupyter notebook password
jupyter notebook --ip $(ip addr | egrep '172\.17|21'| grep 'inet ' | awk '{print $2}' | cut -d/ -f1) --no-browser
```

At the end of the command, you should see a link like this:

```
[I 17:45:05.756 NotebookApp] Serving notebooks from local directory: /mnt/irisgpfs/users/cparisot/Python2
[I 17:45:05.756 NotebookApp] 0 active kernels
[I 17:45:05.756 NotebookApp] The Jupyter Notebook is running at:
[I 17:45:05.757 NotebookApp] http://172.17.6.55:8888/
[I 17:45:05.757 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```

Now we need to create an SSH tunneling in order to access this URL on your laptop. We will forward every local requests made on your localhost address of your laptop the cluster. The simpliest way to do so is to type this command if your are on Linux:

```bash
ssh -NL 8888:<IP OF YOUR NODE>:8888 iris-cluster
# In my example, my command will be: ssh -NL 8888:172.17.6.55:8888 iris-cluster
```

Then, by accessing to your local port 8888 on the localhost address 127.0.0.1, you should see the Python notebook.

[http://127.0.0.1:8888/](http://127.0.0.1:8888/).

If you haven't chosen a password to protect your notebook, please append the token to the URL before accessing your Notebook.

# Run our first notebook

* Just click on the Notebook **jupyter/Monte-Carlo calculation of pi.ipynb**.
* Change the kernel for the `venv` one
  * Go onto **Kernel** tab
  * Choose **Change kernel**
  * Select our previously generated kernel called **venv**
* Try to run the code of the notebook in the kernel by using 'Alt-Enter' keys.
