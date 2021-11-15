[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/tools/easybuild/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/tools/easybuild/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/tools/easybuild/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Using and Building [custom] software with EasyBuild on the UL HPC platform

Copyright (c) 2014-2021 UL HPC Team  <hpc-sysadmins@uni.lu>

Authors: Xavier Besseron, Maxime Schmitt, Sarah Peter, Sébastien Varrette

[![](https://github.com/ULHPC/tutorials/raw/devel/tools/easybuild/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/tools/easybuild/slides.pdf)

The objective of this tutorial is to show how tools such as [EasyBuild](http://easybuild.readthedocs.io) or [stow](https://www.gnu.org/software/stow/)  can be used to ease, automate and script the build of software on the [UL HPC](https://hpc.uni.lu) platforms.

Indeed, as researchers involved in many cutting-edge and hot topics, you probably have access to many theoretical resources to understand the surrounding concepts. Yet it should _normally_ give you a wish to test the corresponding software.
Traditionally, this part is rather time-consuming and frustrating, especially when the developers did not rely on a "regular" building framework such as [CMake](https://cmake.org/) or the [autotools](https://www.gnu.org/software/automake/manual/html_node/Autotools-Introduction.html) (_i.e._ with build instructions as `configure --prefix <path> && make && make install`).

And when it comes to have a build adapted to an HPC system, you are somehow _forced_ to make a custom build performed on the target machine to ensure you will get the best possible performances.
[EasyBuild](https://github.com/easybuilders/easybuild) is one approach to facilitate this step.

Moreover, later on, you probably want to recover a system configuration matching the detailed installation paths through a set of environmental variable  (ex: `JAVA_HOME`, `HADOOP_HOME` etc.). At least you would like to see the traditional `PATH`, `CPATH` or `LD_LIBRARY_PATH` updated.

**Question**: What is the purpose of the above mentioned environmental variable?

For this second aspect, the solution came long time ago (in 1991) with the [Environment Modules](http://modules.sourceforge.net/ and [LMod](https://lmod.readthedocs.io/)
We will cover it in the first part of this tutorial, also to discover the [ULHPC User Software set](https://hpc-docs.uni.lu/environment/modules/) in place.

In a second part, installation using an old yet effective tool named [GNU stow](https://www.gnu.org/software/stow/) will be depicted.

Finally the last part will cover [Easybuild usage on the ULHPC platform](https://hpc-docs.uni.lu/environment/easybuild/) to build and complete the existing software environment.


--------------------
## Pre-requisites ##

Ensure you are able to [connect to the UL HPC clusters](https://hpc.uni.lu/users/docs/access.html).
In particular, recall that the `module` command **is not** available on the access frontends.

If you have never configured [Tmux](https://github.com/tmux/tmux/wiki) or [GNU Screen](http://www.gnu.org/software/screen/) before, and while not strictly mandatory, we advise you to rely on these tools -- see ["HPC Management of Sequential and Embarrassingly Parallel Jobs"](../../sequential/basics/) tutorial.

Now you'll need to pull the latest changes in your working copy of the [ULHPC/tutorials](https://github.com/ULHPC/tutorials) you should have cloned in `~/git/github.com/ULHPC/tutorials` (see ["preliminaries" tutorial](../../preliminaries/))

``` bash
(access)$> cd ~/git/github.com/ULHPC/tutorials
(access)$> git pull
```

Now **configure a dedicated directory `~/tutorials/easybuild` for this session**

``` bash
# return to your home
(access)$> mkdir -p ~/tutorials/easybuild
(access)$> cd ~/tutorials/easybuild
# create a symbolic link to the reference material
(access)$> ln -s ~/git/github.com/ULHPC/tutorials/tools/easybuild ref.d
```

**Advanced users** (_eventually_ yet __strongly__ recommended), create a [Tmux](https://github.com/tmux/tmux/wiki) session (see [Tmux cheat sheet](https://tmuxcheatsheet.com/) and [tutorial](https://www.howtogeek.com/671422/how-to-use-tmux-on-linux-and-why-its-better-than-screen/)) or [GNU Screen](http://www.gnu.org/software/screen/) session you can recover later. See also ["Getting Started" tutorial ](../../beginners/).

``` bash
# /!\ Advanced (but recommended) best-practice:
#     Always work within a TMux or GNU Screen session named '<topic>' (Adapt accordingly)
(access-aion)$> tmux new -s HPC-school   # Tmux
(access-iris)$> screen -S HPC-school     # GNU Screen
#  TMux     | GNU Screen | Action
# ----------|------------|----------------------------------------------
#  CTRL+b c | CTRL+a c   | (create) creates a new Screen window. The default Screen number is zero.
#  CTRL+b n | CTRL+a n   | (next) switches to the next window.
#  CTRL+b p | CTRL+a p   | (prev) switches to the previous window.
#  CTRL+b , | CTRL+a A   | (title) rename the current window
#  CTRL+b d | CTRL+a d   | (detach) detaches from a Screen -
# Once detached:
#   tmux ls  | screen -ls : list available screen
#   tmux att | screen -x  : reattach to a past screen
```

**For all tests and compilation with Easybuild, you MUST work on a computing node.**
Let's get one in interactive jobs:

```bash
### Access to ULHPC cluster (if not yet done)
(laptop)$> ssh aion-cluster
### Have an interactive job, request 4 cores/thread for 2 hours
# ... either directly
(access)$> si --ntasks-per-node 1 -c 4 -t 2:00:00
# ... or using the HPC School reservation 'hpcschool'if needed  - use 'sinfo -T' to check if active and its name
# (access)$> si --reservation=hpcschool --ntasks-per-node 1 -c 4 -t 2:00:00
(node)$>
```

------------------------------------------
## Part 1: Environment modules and LMod ##

The ULHPC facility relied on the [Environment Modules](http://modules.sourceforge.net/) / [LMod](http://lmod.readthedocs.io/en/latest/) framework which provided the [`module`](http://lmod.readthedocs.io) utility **on Compute nodes**
to manage nearly all software.
There are two main advantages of the `module` approach:

1. ULHPC can provide many different versions and/or installations of a
   single software package on a given machine, including a default
   version as well as several older and newer version.
2. Users can easily switch to different versions or installations
   without having to explicitly specify different paths. With modules,
   the `PATH` and related environment variables (`LD_LIBRARY_PATH`, `MANPATH`, etc.) are automatically   managed.

[Environment Modules](http://modules.sourceforge.net/) in itself are a standard and well-established technology across HPC sites, to permit developing and using complex software and libraries build with dependencies, allowing multiple versions of software stacks and combinations thereof to co-exist.
It **brings the `module` command** which is used to manage environment variables such as `PATH`, `LD_LIBRARY_PATH` and `MANPATH`, enabling the easy loading and unloading of application/library profiles and their dependencies.

See <https://hpc-docs.uni.lu/environment/modules/> for more details


| Command                        | Description                                                   |
|--------------------------------|---------------------------------------------------------------|
| `module avail`                 | Lists all the modules which are available to be loaded        |
| `module spider <pattern>`      | Search for <pattern> among available modules **(Lmod only)**  |
| `module load <mod1> [mod2...]` | Load a module                                                 |
| `module unload <module>`       | Unload a module                                               |
| `module list`                  | List loaded modules                                           |
| `module purge`                 | Unload all modules (purge)                                    |
| `module display <module>`      | Display what a module does                                    |
| `module use <path>`            | Prepend the directory to the MODULEPATH environment variable  |
| `module unuse <path>`          | Remove the directory from the MODULEPATH environment variable |

At the heart of environment modules interaction resides the following components:

* the `MODULEPATH` environment variable, which defines the list of searched directories for modulefiles
* `modulefile`

Take a look at the current values:

``` bash
# Example on Aion
$ module -h
$ echo $MODULEPATH
/opt/apps/resif/aion/2020b/epyc/modules/all:/opt/apps/smc/modules
$ module show toolchain/foss
-----------------------------------------------------------------------------------------------------------------
   /opt/apps/resif/aion/2020b/epyc/modules/all/toolchain/foss/2020b.lua:
-----------------------------------------------------------------------------------------------------------------
help([[
Description
===========
GNU Compiler Collection (GCC) based compiler toolchain, including
 OpenMPI for MPI support, OpenBLAS (BLAS and LAPACK support), FFTW and ScaLAPACK.


More information
================
 - Homepage: https://easybuild.readthedocs.io/en/master/Common-toolchains.html#foss-toolchain
]])
whatis("Description: GNU Compiler Collection (GCC) based compiler toolchain, including
 OpenMPI for MPI support, OpenBLAS (BLAS and LAPACK support), FFTW and ScaLAPACK.")
whatis("Homepage: https://easybuild.readthedocs.io/en/master/Common-toolchains.html#foss-toolchain")
whatis("URL: https://easybuild.readthedocs.io/en/master/Common-toolchains.html#foss-toolchain")
conflict("toolchain/foss")
load("compiler/GCC/10.2.0")
load("mpi/OpenMPI/4.0.5-GCC-10.2.0")
load("numlib/OpenBLAS/0.3.12-GCC-10.2.0")
load("numlib/FFTW/3.3.8-gompi-2020b")
load("numlib/ScaLAPACK/2.1.0-gompi-2020b")
setenv("EBROOTFOSS","/opt/apps/resif/aion/2020b/epyc/software/foss/2020b")
setenv("EBVERSIONFOSS","2020b")
setenv("EBDEVELFOSS","/opt/apps/resif/aion/2020b/epyc/software/foss/2020b/easybuild/toolchain-foss-2020b-easybuild-de
vel")
```

You have already access to a huge list of software:

```bash
$ module avail       # OR 'module av'
```

Now you can search for a given software using `module spider <pattern>`:

```bash
$  module spider lang/Python
---------------------------------------------------------------------------------------------------------
  lang/Python:
---------------------------------------------------------------------------------------------------------
    Description:
      Python is a programming language that lets you work more quickly and integrate your systems more
      effectively.

     Versions:
        lang/Python/2.7.18-GCCcore-10.2.0
        lang/Python/3.8.6-GCCcore-10.2.0
```

Let's see the effect of loading/unloading a module

```bash
$> module list
No modules loaded
$> which python
/usr/bin/python
$> python --version       # System level python
Python 2.7.8

$> module load lang/Python    # use TAB to auto-complete
$> which python
/opt/apps/resif/aion/2020b/epyc/software/Python/3.8.6-GCCcore-10.2.0/bin/python
$> python --version
Python Python 3.8.6
$> module purge
```

## ULHPC `$MODULEPATH`

By default, the `MODULEPATH` environment variable holds a _single_ searched directory holding the optimized builds prepared for you by the ULHPC Team.
The general format of this directory is as follows:

```
/opt/apps/resif/<cluster>/<version>/<arch>/modules/all
```

where:

* `<cluster>` depicts the name of the cluster (`iris` or `aion`). Stored as `$ULHPC_CLUSTER`.
* `<version>` corresponds to the ULHPC Software set release (aligned with [Easybuid toolchains release](https://easybuild.readthedocs.io/en/master/Common-toolchains.html#component-versions-in-foss-toolchain)), _i.e._ `2019b`, `2020a` etc. It is stored as `$RESIF_VERSION_{PROD,DEVEL,LEGACY}`.
* `<arch>` is a lower-case strings that categorize the CPU architecture of the build host, and permits to easyli identify optimized target architecture. It is stored as `$RESIF_ARCH`
    - On Intel nodes: `broadwell` (_default_), `skylake`
    - On AMD nodes: `epyc`
    - On GPU nodes: `gpu`

| Cluster                          | Arch.                 | `$MODULEPATH` Environment variable                     |
|----------------------------------|-----------------------|--------------------------------------------------------|
| [Iris](../systems/iris/index.md) | `broadwell` (default) | `/opt/apps/resif/iris/<version>/broadwell/modules/all` |
| [Iris](../systems/iris/index.md) | `skylake`             | `/opt/apps/resif/iris/<version>/skylake/modules/all`   |
| [Iris](../systems/iris/index.md) | `gpu`                 | `/opt/apps/resif/iris/<version>/gpu/modules/all`       |
| [Aion](../systems/aion/index.md) | `epyc` (default)      | `/opt/apps/resif/aion/<version>/{epyc}/modules/all`    |

[![](https://hpc-docs.uni.lu/environment/images/ULHPC-software-stack.png)](https://hpc-docs.uni.lu/environment/modules/#module-naming-schemes)


Now let's assume that a given software you're looking at is not available, or not in the version you want.
Before we continue, there are a set of local environmental variables defined on the ULHPC facility that you will be interested to use in the sequel.

| Environment Variable                 | Description                                                  | Example                                         |
|--------------------------------------|--------------------------------------------------------------|-------------------------------------------------|
| `$ULHPC_CLUSTER`                     | Current ULHPC supercomputer you're connected to              | `aion`                                          |
| `$RESIF_VERSION_{PROD,DEVEL,LEGACY}` | Production / development / legacy ULHPC software set version | `2020b`                                         |
| `$RESIF_ARCH`                        | RESIF Architecture                                           | `epyc`                                          |
| `$MODULEPATH_{PROD,DEVEL,LEGACY}`    | Production / development / legacy MODULEPATH                 | `/opt/apps/resif/aion/2021b/{epyc}/modules/all` |
|                                      |                                                              |                                                 |


## Part 2: GNU Stow (Autotools software)

That's somehow the OLD way yet still very effective way of managing locally built software.
It was used to quickly build the latest version of the [`parallel`](https://www.gnu.org/software/parallel/) command in the ["GNU Parallel" Tutorial](../../sequential/gnu-parallel/).
It assumes to understand the following concepts:

* **stow directory** (`~/stow`): root directory which contains all the stow packages, each with their own private subtree.
    - each subdirectory represents a stow package (`<name>-<version>-<cluster>` typically)
* **stow package**: nothing more than a list of files and directories related to a specific software, managed as an entity.
* **stow target directory**: the directory in which the package files must appear to be installed.
    - By default the stow target directory is considered to be the one _above_ the directory in which stow is invoked from. This behaviour can be easily changed by using the `-t` option (short for `--target`), which allows us to specify an alternative directory.

Stow permits to quickly install / uninstall a stow package as follows:

```bash
cd ~/stow
stow <name>-<version>-<cluster>  # install / enable <name> package
# [...]
stow -D <name>-<version>-<cluster>  # uninstall / disable <name> package
```

You first need to setup your homedir to host stow packages:

```bash
$ cd      # go to your HOME
$ mkdir -p bin include lib share/{doc,man} src
# create stowdir
$ mkdir stow
```

We are going to build and install 2 concurrent versions of a given software available on the ULHPC yet on an old version: [GNU parallel](https://www.gnu.org/software/parallel/).

* build the lastest (up-to-date) version as in the ["GNU Parallel" Tutorial](../../sequential/gnu-parallel/).
* build the version [20201222](https://ftp.gnu.org/gnu/parallel/) version

Now that there are 2 clusters available for production, you are advised to rename stow packages to reflect the cluster on which it was built:

```bash
### Ex: building GNU parallel
# check default (system) verison
$ which parallel
/usr/bin/parallel
$ parallel --version
GNU parallel 20190922

# build latest version under ~/src/parallel/[...]
$ mkdir -p ~/src/parallel
$ cd ~/src/parallel
# download latest version
$ wget https://ftpmirror.gnu.org/parallel/parallel-latest.tar.bz2     # 20211022
$ wget  https://ftpmirror.gnu.org/parallel/parallel-20201222.tar.bz2  # 20201222
# uncompress
$ tar xf parallel-latest.tar.bz2
$ tar xf parallel-20201222.tar.bz2
### Build into a stow package those name reflect the building cluster and/or arch
#  Build latest version...
$ cd parallel-20211022
# abstracted/geeky form to make the prefix directory align to current top directory
#                 if in doubt, use --prefix ~/stow/<name>-<version>-${ULHPC_CLUSTER}
$ ./configure --prefix ~/stow/$(basename $(pwd))-${ULHPC_CLUSTER}
$ make && make install
# ... and intermediate one
$ cd ../parallel-20201222
$ ./configure --prefix ~/stow/$(basename $(pwd))-${ULHPC_CLUSTER}
$ make && make install
```

Now you have 3 versions co-existing:

* the system version `/usr/bin/parallel` (enabled)
* the latest built one as a stow package `~/stow/parallel-20211222-${ULHPC_CLUSTER}/bin/parallel`
* the intermediate built as a stow package `~/stow/parallel-20201222-${ULHPC_CLUSTER}/bin/parallel`

You can quickly enable the latest version with stow:

```
$ cd ~/stow
$ stow parallel-20211222-${ULHPC_CLUSTER}
# check effect
$ ll ~/bin/parallel
lrwxrwxrwx 1 svarrette clusterusers 43 Nov 16 00:43 /home/users/svarrette/bin/parallel -> ../stow/parallel-20211022-aion/bin/parallel
$ which parallel
~/bin/parallel
$ parallel --version
GNU parallel 20211022
```

Imagine that you want to quickly check the intermediate version, with stow it's straightforward:

```bash
$ cd ~/stow
# uninstall/disable stow package -- ignore the 'BUG in find_stowed_path?' message due to https://github.com/aspiers/stow/issues/65
$ stow -D parallel-20211222-${ULHPC_CLUSTER}
$ ll ~/bin/parallel
ls: cannot access /home/users/svarrette/bin/parallel: No such file or directory
$ which parallel
/usr/bin/parallel
$  parallel --version
bash: /home/users/svarrette/bin/parallel: No such file or directory
# occasional error -- refresh your path profile
$ source ~/.profile
$ parallel --version
GNU parallel 20190922
$ Enable intermediate version
$ stow parallel-20201222-aion
$ parallel --version
GNU parallel 20201222
```

GNU stow is very useful for software relying on Autotools (`./configure [...]; make && make install`).
Yet nowadays most software builds are way more complex and rely more and more on over building tools (`cmake`, `ninja`, internal scripts etc.).
You would need a consistent workflow to build the software.
That's where [EasyBuild](http://easybuild.readthedocs.io) comes into play.

-----------------------
## Part 3: Easybuild ##

[<img width='150px' src='http://easybuild.readthedocs.io/en/latest/_static/easybuild_logo_alpha.png'/>](https://easybuilders.github.io/easybuild/)

EasyBuild is a tool that allows to perform automated and reproducible compilation and installation of software. A large number of scientific software are supported (**[2133 supported software packages](http://easybuild.readthedocs.io/en/latest/version-specific/Supported_software.html)** in the last release 4.3.1) -- see also [What is EasyBuild?](http://easybuild.readthedocs.io/en/latest/Introduction.html)

All builds and installations are performed at user level, so you don't need the admin (i.e. `root`) rights.
The software are installed in your home directory (by default in `$HOME/.local/easybuild/software/`) and a module file is generated (by default in `$HOME/.local/easybuild/modules/`) to use the software.

EasyBuild relies on two main concepts: *Toolchains* and *EasyConfig files*.

A **toolchain** corresponds to a compiler and a set of libraries which are commonly used to build a software.
The two main toolchains frequently used on the UL HPC platform are the `foss` ("_Free and Open Source Software_") and the `intel` one.

1. `foss`  is based on the GCC compiler and on open-source libraries (OpenMPI, OpenBLAS, etc.).
2. `intel` is based on the Intel compiler and on Intel libraries (Intel MPI, Intel Math Kernel Library, etc.).

An **EasyConfig file** is a simple text file that describes the build process of a software. For most software that uses standard procedures (like `configure`, `make` and `make install`), this file is very simple.
Many [EasyConfig files](https://github.com/easybuilders/easybuild-easyconfigs/tree/master/easybuild/easyconfigs) are already provided with EasyBuild.
By default, EasyConfig files and generated modules are named using the following convention:
`<Software-Name>-<Software-Version>-<Toolchain-Name>-<Toolchain-Version>`.
However, we use a **hierarchical** approach where the software are classified under a category (or class) -- see  the `CategorizedModuleNamingScheme` option for the `EASYBUILD_MODULE_NAMING_SCHEME` environmental variable), meaning that the layout will respect the following hierarchy:
`<Software-Class>/<Software-Name>/<Software-Version>-<Toolchain-Name>-<Toolchain-Version>`

Additional details are available on EasyBuild website:

- [EasyBuild homepage](https://easybuilders.github.io/easybuild/)
- [EasyBuild documentation](http://easybuild.readthedocs.io/)
- [What is EasyBuild?](http://easybuild.readthedocs.io/en/latest/Introduction.html)
- [Toolchains](https://github.com/easybuilders/easybuild/wiki/Compiler-toolchains)
- [EasyConfig files](http://easybuild.readthedocs.io/en/latest/Writing_easyconfig_files.html)
- [List of supported software packages](http://easybuild.readthedocs.io/en/latest/version-specific/Supported_software.html)

### a. Installation

* [the official instructions](http://easybuild.readthedocs.io/en/latest/Installation.html).

What is important for the installation of Easybuild are the following variables:

* `EASYBUILD_PREFIX`: where to install **local** modules and software, _i.e._ `$HOME/.local/easybuild`
* `EASYBUILD_MODULES_TOOL`: the type of [modules](http://modules.sourceforge.net/) tool you are using, _i.e._ `LMod` in this case
* `EASYBUILD_MODULE_NAMING_SCHEME`: the way the software and modules should be organized (flat view or hierarchical) -- we're advising on `CategorizedModuleNamingScheme`

Add the following entries to your `~/.bashrc` (use your favorite CLI editor like `nano` or `vim`):

```bash
# Easybuild
export EASYBUILD_PREFIX=$HOME/.local/easybuild
export EASYBUILD_MODULES_TOOL=Lmod
export EASYBUILD_MODULE_NAMING_SCHEME=CategorizedModuleNamingScheme
# Use the below variable to run:
#    module use $LOCAL_MODULES
#    module load tools/EasyBuild
export LOCAL_MODULES=${EASYBUILD_PREFIX}/modules/all

alias ma="module avail"
alias ml="module list"
function mu(){
   module use $LOCAL_MODULES
   module load tools/EasyBuild
}
```

Then source this file to expose the environment variables:

```bash
$> source ~/.bashrc
$> echo $EASYBUILD_PREFIX
/home/users/<login>/.local/easybuild
```

Now let's install Easybuild following the [boostrapping procedure](http://easybuild.readthedocs.io/en/latest/Installation.html#bootstrapping-easybuild)

**`/!\ IMPORTANT:`**  Recall that **you should be on a compute node to install [Easybuild]((http://easybuild.readthedocs.io)** (otherwise the checks of the `module` command availability will fail.)

A dedicated script [`scripts/setup.sh`](https://github.com/ULHPC/tutorials/blob/devel/tools/easybuild/scripts/setup.sh) has been prepared to facilitate the install/update of your local version of Easybuild:

```bash
### Access to ULHPC cluster (if not yet done)
(laptop)$> ssh iris-cluster
# Have an interactive job (if not yet done)
(access-iris)$> si
# Run the setup script
(node)$> ./scripts/setup.sh -h  # help
(node)$> ./scripts/setup.sh -n  # Dry-run
(node)$> ./scripts/setup.sh
```
This script basically perform the following tasks:

``` bash
# download script
(node)$> curl -o /tmp/bootstrap_eb.py https://raw.githubusercontent.com/easybuilders/easybuild-framework/develop/easybuild/scripts/bootstrap_eb.py

# install Easybuild
(node)$> echo $EASYBUILD_PREFIX
/home/users/<login>/.local/easybuild
(node)$> python /tmp/bootstrap_eb.py $EASYBUILD_PREFIX
```

Now you can use your freshly built software. The main EasyBuild command is `eb`:

```bash
(node)$> eb --version             # expected ;)
-bash: eb: command not found

# Load the newly installed Easybuild
(node)$> echo $MODULEPATH
/opt/apps/resif/iris/2019b/broadwell/modules/all

(node)$> module use $LOCAL_MODULES
(node)$> echo $MODULEPATH
/home/users/<login>/.local/easybuild/modules/all:/opt/apps/resif/iris/2019b/broadwell/modules/all

(node)$> module spider Easybuild
(node)$> module load tools/EasyBuild       # TAB is your friend...
(node)$> eb --version
This is EasyBuild 4.3.2 (framework: 4.3.2, easyblocks: 4.3.2) on host iris-080.
```

Since you are going to use quite often the above command to use locally built modules and load easybuild, an alias `mu` is provided and can be used from now on. Use it **now**.

```
(node)$> mu
(node)$> module avail     # OR 'ma'
```
To get help on the EasyBuild options, use the `-h` or `-H` option flags:

    (node)$> eb -h
    (node)$> eb -H

### b. Local vs. global usage

As you probably guessed, we are going to use two places for the installed software:

* local builds `~/.local/easybuild`          (see `$LOCAL_MODULES`)
* global builds (provided to you by the UL HPC team) in `/opt/apps/resif/data/stable/default/modules/all` (see default `$MODULEPATH`).

Default usage (with the `eb` command) would install your software and modules in `~/.local/easybuild`.

Before that, let's explore the basic usage of [EasyBuild](http://easybuild.readthedocs.io/) and the `eb` command.

```bash
(node)$> module av Tensorflow
----------------- /opt/apps/resif/iris/2019b/broadwell/modules/all -----------------------------
   lib/TensorFlow/2.1.0-foss-2019b-Python-3.7.4
   tools/Horovod/0.19.1-foss-2019b-TensorFlow-2.1.0-Python-3.7.4

# Search for an Easybuild recipy with 'eb -S <pattern>'
(node)$>  eb -S Tensorflow
CFGS1=/opt/apps/resif/data/easyconfigs/ulhpc/iris/easybuild/easyconfigs
CFGS2=/opt/apps/resif/data/easyconfigs/ulhpc/default/easybuild/easyconfigs
CFGS3=/home/users/svarrette/.local/easybuild/software/EasyBuild/4.3.1/easybuild/easyconfigs
[...]
 * $CFGS3/t/TensorFlow/TensorFlow-2.2.0-foss-2019b-Python-3.7.4.eb
 * $CFGS3/t/TensorFlow/TensorFlow-2.2.0-fosscuda-2019b-Python-3.7.4.eb
 * $CFGS3/t/TensorFlow/TensorFlow-2.3.1-foss-2019b-Python-3.7.4.eb
 * $CFGS3/t/TensorFlow/TensorFlow-2.3.1-foss-2020a-Python-3.8.2.eb
 * $CFGS3/t/TensorFlow/TensorFlow-2.3.1-fosscuda-2019b-Python-3.7.4.eb
```

We can see that in this example, a more recent version of Tensorflow exists, matching the 2019b toolchain.


### c. Build software using provided EasyConfig file

In this part, we propose to build a more recent version of [Tensorflow](https://www.tensorflow.org/) using EasyBuild.
As seen above, we are going to build Tensorflow 2.3.1  against the `foss` toolchain, typically the 2019b version which is available by default on the platform.

Pick the corresponding recipy (for instance `TensorFlow-2.3.1-foss-2019b-Python-3.7.4.eb`), install it with

       eb <name>.eb [-D] -r

* `-D` enables the dry-run mode to check what's going to be install -- **ALWAYS try it first**
* `-r` enables the robot mode to automatically install all dependencies while searching for easyconfigs in a set of pre-defined directories -- you can also prepend new directories to search for eb files (like the current directory `$PWD`) using the option and syntax `--robot-paths=$PWD:` (do not forget the ':'). See [Controlling the robot search path documentation](http://easybuild.readthedocs.io/en/latest/Using_the_EasyBuild_command_line.html#controlling-the-robot-search-path)
* The `$CFGS<n>/` prefix should be dropped unless you know what you're doing (and thus have previously defined the variable -- see the first output of the `eb -S [...]` command).

So let's install `Tensorflow` version 2.3.1 and **FIRST** check which dependencies are satisfied with `-Dr`:

```bash
(node)$> eb TensorFlow-2.3.1-foss-2019b-Python-3.7.4.eb -Dr
== temporary log file in case of crash /tmp/eb-xlOj8P/easybuild-z4CDzy.log
== found valid index for /home/users/svarrette/.local/easybuild/software/EasyBuild/4.3.1/easybuild/easyconfigs, so using it...
Dry run: printing build status of easyconfigs and dependencies
* [x] $CFGS/m/M4/M4-1.4.18.eb (module: devel/M4/1.4.18)
* [x] $CFGS/j/Java/Java-1.8.0_241.eb (module: lang/Java/1.8.0_241)
* [x] $CFGS/j/Java/Java-1.8.eb (module: lang/Java/1.8)
[...]
* [x] $CFGS/p/pkg-config/pkg-config-0.29.2-GCCcore-8.3.0.eb (module: devel/pkg-config/0.29.2-GCCcore-8.3.0)
* [ ] $CFGS/d/DB/DB-18.1.32-GCCcore-8.3.0.eb (module: tools/DB/18.1.32-GCCcore-8.3.0)
* [x] $CFGS/g/giflib/giflib-5.2.1-GCCcore-8.3.0.eb (module: lib/giflib/5.2.1-GCCcore-8.3.0)
[...]
* [x] $CFGS/i/ICU/ICU-64.2-GCCcore-8.3.0.eb (module: lib/ICU/64.2-GCCcore-8.3.0)
* [ ] $CFGS/b/Bazel/Bazel-3.4.1-GCCcore-8.3.0.eb (module: devel/Bazel/3.4.1-GCCcore-8.3.0)
* [x] $CFGS/g/git/git-2.23.0-GCCcore-8.3.0-nodocs.eb (module: tools/git/2.23.0-GCCcore-8.3.0-nodocs)
* [x] $CFGS/s/SWIG/SWIG-4.0.1-GCCcore-8.3.0.eb (module: devel/SWIG/4.0.1-GCCcore-8.3.0)
 * [ ] $CFGS/t/TensorFlow/TensorFlow-2.3.1-foss-2019b-Python-3.7.4.eb (module: lib/TensorFlow/2.3.1-foss-2019b-Python-3.7.4)
== Temporary log file(s) /tmp/eb-xlOj8P/easybuild-z4CDzy.log* have been removed.
== Temporary directory /tmp/eb-xlOj8P has been removed.
```

As can be seen, there was a few elements to install and this has not been done so far (box not checked). Most of the dependencies are already present (box checked).
Let's really install the selected software -- you may want to prefix the `eb` command with the `time` to collect the installation time:

```bash
(node)$> eb TensorFlow-2.3.1-foss-2019b-Python-3.7.4.eb -r
== temporary log file in case of crash /tmp/eb-tqZXLe/easybuild-wJX_gs.log
== found valid index for /home/users/svarrette/.local/easybuild/software/EasyBuild/4.3.1/easybuild/easyconfigs, so using it...
== resolving dependencies ...
== processing EasyBuild easyconfig /home/users/svarrette/.local/easybuild/software/EasyBuild/4.3.1/easybuild/easyconfigs/b/Bazel/Bazel-3.4.1-GCCcore-8.3.0.eb
== building and installing devel/Bazel/3.4.1-GCCcore-8.3.0...
== fetching files...
== creating build dir, resetting environment...
== unpacking...
== patching...
== preparing...
== configuring...
== building...
== testing...
== installing...
== taking care of extensions...
== restore after iterating...
== postprocessing...
== sanity checking...
== cleaning up...
== creating module...
== permissions...
== packaging...
== COMPLETED: Installation ended successfully (took 3 min 26 sec)
== Results of the build can be found in the log file(s) /home/users/svarrette/.local/easybuild/software/Bazel/3.4.1-GCCcore-8.3.0/easybuild/easybuild-Bazel-3.4.1-20201209.215735.log
== processing EasyBuild easyconfig /home/users/svarrette/.local/easybuild/software/EasyBuild/4.3.1/easybuild/easyconfigs/t/TensorFlow/TensorFlow-2.3.1-foss-2019b-Python-3.7.4.eb
== building and installing lib/TensorFlow/2.3.1-foss-2019b-Python-3.7.4...
== fetching files...
== creating build dir, resetting environment...
== unpacking...
== patching...
== preparing...
== configuring...
== building...
== testing...
== installing...
== taking care of extensions...
== installing extension Markdown 3.2.2 (1/23)...
== installing extension pyasn1-modules 0.2.8 (2/23)...
== installing extension rsa 4.6 (3/23)...
[...]
= installing extension Keras-Preprocessing 1.1.2 (22/23)...
== installing extension TensorFlow 2.3.1 (23/23)...
== restore after iterating...
== postprocessing...
== sanity checking...
== cleaning up...
== creating module...
== permissions...
== packaging...
== COMPLETED: Installation ended successfully (took 38 min 47 sec)
[...]
```

Check the installed software:

```
(node)$> module av Tensorflow

------------------------- /home/users/<login>/.local/easybuild/modules/all -------------------------
   lib/TensorFlow/2.3.1-foss-2019b-Python-3.7.4

----------------- /opt/apps/resif/iris/2019b/broadwell/modules/all -----------------------------
   lib/TensorFlow/2.1.0-foss-2019b-Python-3.7.4
   tools/Horovod/0.19.1-foss-2019b-TensorFlow-2.1.0-Python-3.7.4



Use "module spider" to find all possible modules.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".
```

**Note**: to see the (locally) installed software, the `MODULEPATH` variable should include the `$HOME/.local/easybuild/modules/all/` (of `$LOCAL_MODULES`) path (which is what happens when using `module use <path>` -- see the `mu` command)

You can now load the freshly installed module like any other:

```bash
(node)$> module load  lib/TensorFlow/2.3.1-foss-2019b-Python-3.7.4
(node)$> module list

Currently Loaded Modules:
  1) tools/EasyBuild/3.6.1                          7) mpi/impi/2017.1.132-iccifort-2017.1.132-GCC-6.3.0-2.27
  2) compiler/GCCcore/6.3.0                         8) toolchain/iimpi/2017a
  3) tools/binutils/2.27-GCCcore-6.3.0              9) numlib/imkl/2017.1.132-iimpi-2017a
  4) compiler/icc/2017.1.132-GCC-6.3.0-2.27        10) toolchain/intel/2017a
  5) compiler/ifort/2017.1.132-GCC-6.3.0-2.27      11) tools/HPL/2.2-intel-2017a
  6) toolchain/iccifort/2017.1.132-GCC-6.3.0-2.27

# Check version
(node)$> python -c 'import tensorflow as tf; print(tf.__version__)'
```

**Tips**: When you load a module `<NAME>` generated by Easybuild, it is installed within the directory reported by the `$EBROOT<NAME>` variable.
In the above case, you will find the generated binary for Tensorflow in `${EBROOTTENSORFLOW}/`.


### d. Build software using a customized EasyConfig file

There are multiple ways to amend an EasyConfig file. Check the `--try-*` option flags for all the possibilities.

Generally you want to do that when the up-to-date version of the software you want is **not** available as a recipy within Easybuild.
For instance, a very popular building environment [CMake](https://blog.kitware.com/cmake-3-19-1-available-for-download/) has recently released a new version (3.19.1), which you want to give a try.

It is not available as module, so let's build it.

First let's check for available easyconfigs recipy if one exist for the expected version:

```
(node)$> eb -S Cmake-3
[...]
 * $CFGS2/c/CMake/CMake-3.12.1.eb
 * $CFGS2/c/CMake/CMake-3.15.3-GCCcore-8.3.0.eb
 * $CFGS2/c/CMake/CMake-3.15.3-fix-toc-flag.patch
 * $CFGS2/c/CMake/CMake-3.16.4-GCCcore-9.3.0.eb
 * $CFGS2/c/CMake/CMake-3.18.4-GCCcore-10.2.0.eb
[...]
```

You may want to reuse the helper script `./scripts/suggest-easyconfigs` to find the versions available (and detect the dependencies version to be place in the custom Easyconfig).

We are going to reuse one of the latest EasyConfig available, for instance lets copy `$CFGS2/c/CMake/CMake-3.18.4-GCCcore-10.2.0.eb` as it was the most recent.
We'll have to make it match the toolchain/compiler available by default in 2019b i.e. GCCcore-8.3.0.

```bash
# Work in a dedicated directory
(node)$> mkdir -p ~/software/CMake
(node)$> cd ~/software/CMake

(node)$> eb -S Cmake-3|less   # collect the definition of the CFGS2 variable
(node)$> CFGS2=/Users/svarrette/git/github.com/ULHPC/easybuild-easyconfigs/easybuild/easyconfigs
(node)$> cp $CFGS2/c/CMake/CMake-3.18.4-GCCcore-10.2.0.eb .
# Adapt the filename with the target version and your default building environement - here 2019b software set
(node)$> mv CMake-3.18.4-GCCcore-8.3.0.eb        # Adapt version suffix to the lastest realse
```

You need to perform the following changes (here: version upgrade, adapted checksum)
To find the appropriate version for the dependencies, use:

``` bash
# Summarize matchin versions for list of dependencies
./scripts/suggest-easyconfigs -s ncurses zlib bzip2 cURL libarchive
           ncurses: ncurses-6.1-GCCcore-8.3.0.eb
              zlib: zlib-1.2.11-GCCcore-8.3.0.eb
             bzip2: bzip2-1.0.8-GCCcore-8.3.0.eb
              cURL: cURL-7.66.0-GCCcore-8.3.0.eb
        libarchive: libarchive-3.4.3-GCCcore-10.2.0.eb
```


```diff
--- CMake-3.18.4-GCCcore-10.2.0.eb	2020-12-09 22:33:12.375199000 +0100
+++ CMake-3.19.1-GCCcore-8.3.0.eb	2020-12-09 22:42:40.238721000 +0100
@@ -1,5 +1,5 @@
 name = 'CMake'
-version = '3.18.4'
+version = '3.19.1'

 homepage = 'https://www.cmake.org'

@@ -8,22 +8,22 @@
  tools designed to build, test and package software.
 """

-toolchain = {'name': 'GCCcore', 'version': '10.2.0'}
+toolchain = {'name': 'GCCcore', 'version': '8.3.0'}

 source_urls = ['https://www.cmake.org/files/v%(version_major_minor)s']
 sources = [SOURCELOWER_TAR_GZ]
-checksums = ['597c61358e6a92ecbfad42a9b5321ddd801fc7e7eca08441307c9138382d4f77']
+checksums = ['1d266ea3a76ef650cdcf16c782a317cb4a7aa461617ee941e389cb48738a3aba']

 builddependencies = [
-    ('binutils', '2.35'),
+    ('binutils', '2.32'),
 ]

 dependencies = [
-    ('ncurses', '6.2'),
+    ('ncurses', '6.1'),
     ('zlib', '1.2.11'),
     ('bzip2', '1.0.8'),
-    ('cURL', '7.72.0'),
-    ('libarchive', '3.4.3'),
+    ('cURL', '7.66.0'),
+    ('libarchive', '3.4.0'),
     # OS dependency should be preferred if the os version is more recent then this version,
     # it's nice to have an up to date openssl for security reasons
     # ('OpenSSL', '1.1.1h'),
```

libarchive will have also to be adapted.

If the checksum is not provided on the [official software page](https://cmake.org/download/), you will need to compute it yourself by downloading the sources and collect the checksum:

```bash
(laptop)$> sha256sum ~/Downloads/cmake-3.19.1.tar.gz
1d266ea3a76ef650cdcf16c782a317cb4a7aa461617ee941e389cb48738a3aba  /Users/svarrette/Downloads/cmake-3.19.1.tar.gz
```

You can now build it

```bash
(node)$> eb ./CMake-3.19.1-GCCcore-8.3.0.eb -Dr
(node)$> eb ./CMake-3.19.1-GCCcore-8.3.0.eb -r
```

**Note** you can follow the progress of the installation in a separate shell on the node:

* (eventually) connect to the allocated node (using `ssh` or `oarsub -C <jobid>` depending on the cluster)
* run `htop`
    - press 'u' to filter by process owner, select your login
    - press 'F5' to enable the tree view

Check the result:

```bash
(node)$> module av CMake
```

That's all ;-)

**Final remaks**

This workflow (copying an existing recipy, adapting the filename, the version and the source checksum) covers most of the test cases.
Yet sometimes you need to work on a more complex dependency check, in which case you'll need to adapt _many_ eb files.
In this case, for each build, you need to instruct Easybuild to search for easyconfigs also in the current directory, in which case you will use:

```bash
$> eb <filename>.eb --robot=$PWD:$EASYBUILD_ROBOT -D
$> eb <filename>.eb --robot=$PWD:$EASYBUILD_ROBOT
```

# Submitting working Easyconfigs to easybuilders

* Follow the __Official documentations__:
    - [Integration with Github](https://easybuild.readthedocs.io/en/latest/Integration_with_GitHub.html)
    - [Submitting pull requests (`--new-pr`)](https://easybuild.readthedocs.io/en/latest/Integration_with_GitHub.html#submitting-pull-requests-new-pr)
    - [Uploading test reports (`--upload-test-report`)](https://easybuild.readthedocs.io/en/latest/Integration_with_GitHub.html#uploading-test-reports-upload-test-report)
    - [Updating existing pull requests (`--update-pr`)](https://easybuild.readthedocs.io/en/master/Integration_with_GitHub.html#updating-existing-pull-requests-update-pr)


## To go further (to update)

- [EasyBuild homepage](http://easybuilders.github.io/easybuild)
- [EasyBuild documentation](http://easybuilders.github.io/easybuild/)
- [Getting started](https://github.com/easybuilders/easybuild/wiki/Getting-started)
- [Using EasyBuild](https://github.com/easybuilders/easybuild/wiki/Using-EasyBuild)
- [Step-by-step guide](https://github.com/easybuilders/easybuild/wiki/Step-by-step-guide)
