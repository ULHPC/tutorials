[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/tools/easybuild/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/tools/easybuild/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/tools/easybuild/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Building [custom] software with EasyBuild on the UL HPC platform

Copyright (c) 2014-2020 UL HPC Team  <hpc-sysadmins@uni.lu>

Authors: Xavier Besseron, Maxime Schmitt, Sarah Peter, Sébastien Varrette

[![](https://github.com/ULHPC/tutorials/raw/devel/tools/easybuild/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/tools/easybuild/slides.pdf)

The objective of this tutorial is to show how [EasyBuild](http://easybuild.readthedocs.io) can be used to ease, automate and script the build of software on the [UL HPC](https://hpc.uni.lu) platforms.

Indeed, as researchers involved in many cutting-edge and hot topics, you probably have access to many theoretical resources to understand the surrounding concepts. Yet it should _normally_ give you a wish to test the corresponding software.
Traditionally, this part is rather time-consuming and frustrating, especially when the developers did not rely on a "regular" building framework such as [CMake](https://cmake.org/) or the [autotools](https://www.gnu.org/software/automake/manual/html_node/Autotools-Introduction.html) (_i.e._ with build instructions as `configure --prefix <path> && make && make install`).

And when it comes to have a build adapted to an HPC system, you are somehow _forced_ to make a custom build performed on the target machine to ensure you will get the best possible performances.
[EasyBuild](https://github.com/easybuilders/easybuild) is one approach to facilitate this step.

Moreover, later on, you probably want to recover a system configuration matching the detailed installation paths through a set of environmental variable  (ex: `JAVA_HOME`, `HADOOP_HOME` etc.). At least you would like to see the traditional `PATH`, `CPATH` or `LD_LIBRARY_PATH` updated.

**Question**: What is the purpose of the above mentioned environmental variable?

For this second aspect, the solution came long time ago (in 1991) with the [Environment Modules](http://modules.sourceforge.net/).
We will cover it in the first part of this tutorial.

Then, another advantage of [EasyBuild](http://easybuild.readthedocs.io) comes into account that justifies its wide-spread deployment across many HPC centers (incl. [UL HPC](http://hpc.uni.lu)): it has been designed to not only build any piece of software, but also to generate the corresponding module files to facilitate further interactions with it.
Thus we will cover [EasyBuild](http://easybuild.readthedocs.io) in the second part of this hands-on.
It allows for automated and reproducable builds of software. Once a build has been made, the build script (via the *EasyConfig file*) or the installed software (via the *module file*) can be shared with other users.

You might be interested to know that we rely on [EasyBuild](http://easybuild.readthedocs.io) to provide the [software environment](https://hpc.uni.lu/users/software/) to the users of the platform.

In this tutorial, we are going to first build software that are supported by EasyBuild. Then we will see through a simple example how to add support for a new software in EasyBuild, and eventually contribute back to the main repository

**Note**: The latest version of this tutorial is available on [Github](https://github.com/ULHPC/tutorials/tree/devel/tools/easybuild/).

--------------------
## Pre-requisites ##

Ensure you are able to [connect to the UL HPC clusters](https://hpc.uni.lu/users/docs/access.html).
**For all tests and compilation with Easybuild, you MUST work on a computing node.**

In particular, the `module` command **is not** available on the access frontends.

```bash
# /!\ FOR ALL YOUR COMPILING BUSINESS, ENSURE YOU WORK ON A COMPUTING NODE
# Advanced users:
#    always work within an GNU Screen session named with 'screen -S <topic>' (Adapt accordingly)
# Have an interactive job
############### iris cluster (slurm) ###############
(access-iris)$> si --ntasks-per-node 1 -c 4 -t 2:00:00
# srun -p interactive --qos debug -C batch --ntasks-per-node 1 -c 4 -t 2:00:00 --mem-per-cpu 4096 --pty bash
(node)$>
```

------------------------------------------
## Part 1: Environment modules and LMod ##

[Environment Modules](http://modules.sourceforge.net/) are a standard and well-established technology across HPC sites, to permit developing and using complex software and libraries build with dependencies, allowing multiple versions of software stacks and combinations thereof to co-exist.

The tool in itself is used to manage environment variables such as `PATH`, `LD_LIBRARY_PATH` and `MANPATH`, enabling the easy loading and unloading of application/library profiles and their dependencies.

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


*Note:* for more information, see the reference man pages for [modules](http://modules.sourceforge.net/man/module.html) and [modulefile](http://modules.sourceforge.net/man/modulefile.html), or the [official FAQ](http://sourceforge.net/p/modules/wiki/FAQ/).

You can also see our [modules page](https://hpc.uni.lu/users/docs/modules.html) on the [UL HPC website](http://hpc.uni.lu/users/).

At the heart of environment modules interaction resides the following components:

* the `MODULEPATH` environment variable, which defines the list of searched directories for modulefiles
* `modulefile` (see [an example](http://www.nersc.gov/assets/modulefile-example.txt)) associated to each available software.

Then, [Lmod](https://www.tacc.utexas.edu/research-development/tacc-projects/lmod)  is a [Lua](http://www.lua.org/)-based module system that easily handles the `MODULEPATH` hierarchical problem.

Lmod is a new implementation of Environment Modules that easily handles the MODULEPATH hierarchical problem. It is a drop-in replacement for TCL/C modules and reads TCL modulefiles directly.
In particular, Lmod adds many interesting features on top of the traditional implementation focusing on an easier interaction (search, load etc.) for the users. Thus that is the tool we would advise to deploy.

* [User guide](https://www.tacc.utexas.edu/research-development/tacc-projects/lmod/user-guide)
* [Advanced user guide](https://www.tacc.utexas.edu/research-development/tacc-projects/lmod/advanced-user-guide)
* [Sysadmins Guide](https://www.tacc.utexas.edu/research-development/tacc-projects/lmod/system-administrators-guide)

**`/!\ IMPORTANT:` (reminder): the `module` command is ONLY available on the nodes, NOT on the access front-ends.**

```bash
$> module -h
$> echo $MODULEPATH
/opt/apps/resif/iris/2019b/broadwell/modules/all
# If you want to force a certain version
unset MODULEPATH
module use /opt/apps/resif/iris/2019b/broadwell/modules/all
```

You have already access to a huge list of software:

```bash
$> module avail       # OR 'module av'
```

Now you can search for a given software using `module spider <pattern>`:

```
$> module spider lang/python

-----------------------------------------------------------------------------------------------------------------
  lang/Python:
-----------------------------------------------------------------------------------------------------------------
    Description:
      Python is a programming language that lets you work more quickly and integrate your systems more effectively.

     Versions:
        lang/Python/2.7.16-GCCcore-8.3.0
        lang/Python/3.7.4-GCCcore-8.3.0
```

Let's see the effect of loading/unloading a module

```bash
$> module list
No modules loaded
$> which python
/usr/bin/python
$> python --version       # System level python
Python 2.7.5

$> module load lang/Python    # use TAB to auto-complete
$> which python
/opt/apps/resif/iris/2019b/broadwell/software/Python/3.7.4-GCCcore-8.3.0/bin/python
$> python --version
Python 3.7.4

$> module purge
```

Now let's assume that a given software you're looking at is not available, or not in the version you want.
That's where [EasyBuild](http://easybuild.readthedocs.io) comes into play.

-----------------------
## Part 2: Easybuild ##

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

**`/!\ IMPORTANT:`**  Recall that **you should be on a compute node to install [Easybuild]((http://easybuild.readthedocs.io)** (otherwise the checks of the `module` command availability will fail  -- it is normally still the case (Use `si -t 02:00:00 -N 1 --ntasks-per-node 1 -c 28` otherwise)

```bash
### Access to ULHPC cluster (if not yet done)
(laptop)$> ssh iris-cluster
# Have an interactive job (if not yet done)
(access-iris)$> si
(node)$> cd
# download script
(node)$> curl -o bootstrap_eb.py  https://raw.githubusercontent.com/easybuilders/easybuild-framework/develop/easybuild/scripts/bootstrap_eb.py

# install Easybuild
(node)$> echo $EASYBUILD_PREFIX
/home/users/<login>/.local/easybuild
(node)$> python bootstrap_eb.py $EASYBUILD_PREFIX
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
This is EasyBuild 4.3.1 (framework: 4.3.1, easyblocks: 4.3.1) on host iris-095.
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
