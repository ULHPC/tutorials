First of all, ensure you have installed the [Pre-requisites / Preliminary software](preliminaries.md) and followed the corresponding configuration.

Then this repository is hosted on [Github](https://github.com/ULHPC/tutorials).
Assuming you have installed and configured `git`:

## Cloning the ULHPC git repository

To clone this repository, proceed as follows (adapt accordingly):

```bash
$> mkdir -p ~/git/github.com/ULHPC
$> cd ~/git/github.com/ULHPC
$> git clone https://github.com/ULHPC/tutorials.git
$> cd tutorials
# /!\ IMPORTANT: run 'make setup' only **AFTER** Pre-requisites software are installed
```

To setup you local copy of this repository (**after** pre-requisites are satisfied), simply run:

```bash
$> make setup    # Under Mac OS / Linux
```

This will initiate the Git submodules of this repository (see `.gitmodules`) and setup the [git flow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) layout for this repository. Later on, you can update your local branches by running:

     $> make up

If upon pulling the repository, you end in a state where another collaborator have upgraded the Git submodules for this repository, you'll end in a dirty state (as reported by modifications within the `.submodules/` directory). In that case, just after the pull, you **have to run** `make up` to ensure consistency with regards the Git submodules.

Finally, you can upgrade the Git submodules to the latest version by running:

    $> make upgrade


## Python Virtualenv / Pyenv and Direnv

If you want to perform a local rendering of this documentation, a few Python packages are required to be installed (see `requirements.txt`).
You will have to ensure you have installed [direnv](https://direnv.net/) (configured by [`.envrc`](.envrc)), [pyenv](https://github.com/pyenv/pyenv) and [`pyenv-virtualenv`](https://github.com/pyenv/pyenv-virtualenv). This assumes also the presence of `~/.config/direnv/direnvrc` from [this page](https://github.com/Falkor/dotfiles/blob/master/direnv/direnvrc) - for more details, see [this blog post](https://varrette.gforge.uni.lu/blog/2019/09/10/using-pyenv-virtualenv-direnv/).

You can run the following command to setup your local machine in a compliant way:

```
make setup-direnv
make setup-pyenv
```

Adapt your favorite shell configuration as suggested. You may want to add the following:

``` bash
for f in $XDG_CONFIG_HOME/*/init.sh; do
  . ${f}
done
```

Running `direnv allow` (this will have to be done only once), you should automatically enable the virtualenv `ulhpc-docs` based on the python version specified in [`.python-version`](.python-version). You'll eventually need to install the appropripriate Python version with `pyenv`:

```bash
pyenv versions   # Plural: show all versions
pyenv install $(head .python-version)
# Activate the virtualenv by reentering into the directory
cd ..
cd -
```

From that point, you should install the required packages using:

``` bash
make setup-python

# OR (manually)
pip install --upgrade pip
pip install -r requirements.txt
```

You should now be able to preview the documentation **locally** (as rendered on [readthedocs](https://ulhpc-tutorials.readthedocs.io/)).

## Documentation

See `docs/`.

The documentation for this project is handled by [`mkdocs`](http://www.mkdocs.org/#installation).
You might wish to generate locally the docs:

* Install [`mkdocs`](http://www.mkdocs.org/#installation) and the mandatory package from the `requirements.txt` file (ideally within a virtual environment as above)
* Preview your documentation from the project root by running `mkdocs serve` and visit with your favorite browser the URL `http://localhost:8000`
     - Alternatively, you can run `make doc` at the root of the repository.
* (eventually) build the full documentation locally (in the `site/` directory) by running `mkdocs build`.

## Prepare for Tutorial sessions

To take the best out the tutorial sessions proposed during the HPC school, you probably wish on your homedir on the cluster to

1. clone (or update) the [`ULHPC/tutorials`](https://github.com/ULHPC/tutorials/) as instructed above
2. work in a separate directory structure when following a given event. Here is a suggested approach:

```bash
# First time: clone the repo under a meaningfull path
$> mkdir -p ~/git/github.com/ULHPC
$> cd ~/git/github.com/ULHPC
$> git clone https://github.com/ULHPC/tutorials.git
$> cd tutorials
$> make setup

# Next times: pull latest changes
$> cd ~/git/github.com/ULHPC/tutorials
$> make up     # update both branches (production and devel)

# Prepare a dedicated (separated) working directory
$> mkdir -p ~/tutorials/ULHPC-School-2020         # Adapt event name accordingly
$> cd ~/tutorials/ULHPC-School-2020
$> ln -s ~/git/github.com/ULHPC/tutorials  ref.d  # create a symbolic link pointing to the tutorial reference material
# Now  $HOME/tutorials/ULHPC-School-2020/ref.d/ points to reference training material
```
