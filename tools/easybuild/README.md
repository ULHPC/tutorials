[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/tools/easybuild/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/tools/easybuild/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/tools/easybuild/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Using and Building [custom] software with EasyBuild on the UL HPC platform

Copyright (c) 2014-2021 UL HPC Team  <hpc-sysadmins@uni.lu>

Authors: Emmanuel Kieffer, Sébastien Varrette and UL HPC Team <hpc-team@uni.lu>

The objective of this tutorial is to show how tools such as [EasyBuild](http://easybuild.readthedocs.io)  can be used to ease, automate and script the build of software on the [UL HPC](https://hpc.uni.lu) platforms.

Indeed, as researchers involved in many cutting-edge and hot topics, you probably have access to many theoretical resources to understand the surrounding concepts. Yet it should _normally_ give you a wish to test the corresponding software.
Traditionally, this part is rather time-consuming and frustrating, especially when the developers did not rely on a "regular" building framework such as [CMake](https://cmake.org/) or the [autotools](https://www.gnu.org/software/automake/manual/html_node/Autotools-Introduction.html) (_i.e._ with build instructions as `configure --prefix <path> && make && make install`).

And when it comes to have a build adapted to an HPC system, you are somehow _forced_ to make a custom build performed on the target machine to ensure you will get the best possible performances.
[EasyBuild](https://github.com/easybuilders/easybuild) is one approach to facilitate this step.

Moreover, later on, you probably want to recover a system configuration matching the detailed installation paths through a set of environmental variable  (ex: `JAVA_HOME`, `HADOOP_HOME` etc.). At least you would like to see the traditional `PATH`, `CPATH` or `LD_LIBRARY_PATH` updated.

**Question**: What is the purpose of the above mentioned environmental variable?

For this second aspect, the solution came long time ago (in 1991) with the [Environment Modules](http://modules.sourceforge.net/ and [LMod](https://lmod.readthedocs.io/)
We will cover it in the first part of this tutorial, also to discover the [ULHPC User Software set](https://hpc-docs.uni.lu/environment/modules/) in place.

Finally the last part will cover [Easybuild usage on the ULHPC platform](https://hpc-docs.uni.lu/environment/easybuild/) to build and complete the existing software environment.


--------------------
## Pre-requisites ##

Ensure you are able to [connect to the UL HPC clusters](https://hpc-docs.uni.lu/connect/access/).

In particular, recall that the `module` command **is not** available on the access frontends.


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
(access)$> ln -s ~/git/github.com/ULHPC/tutorials/tools/easybuild easybuild
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
* `<arch>` is a lower-case strings that categorize the CPU architecture of the build host, and permits to easily identify optimized target architecture. It is stored as `$RESIF_ARCH`
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



-----------------------
## Part 2: Easybuild ##

[<img width='150px' src='https://easybuild.io/images/easybuild_logo_horizontal.png'/>](https://easybuilders.github.io/easybuild/)

EasyBuild is a tool that allows to perform automated and reproducible compilation and installation of software. A large number of scientific software are supported (**[2995 supported software packages](http://easybuild.readthedocs.io/en/latest/version-specific/Supported_software.html)** in the last release 4.7.1) -- see also [What is EasyBuild?](http://easybuild.readthedocs.io/en/latest/Introduction.html)

All builds and installations are performed at user level, so you don't need the admin (i.e. `root`) rights.
The software are installed in your home directory under `$EASYBUILD_PREFIX` -- see <https://hpc-docs.uni.lu/environment/easybuild/>

|                     | Default setting (local)  | Recommended setting                                                           |
|---------------------|--------------------------|-------------------------------------------------------------------------------|
| `$EASYBUILD_PREFIX` | `$HOME/.local/easybuild` | `$HOME/.local/easybuild/${ULHPC_CLUSTER}/${RESIF_VERSION_PROD}/${RESIF_ARCH}` |
|                     |                          |                                                                               |

* built software are placed under `${EASYBUILD_PREFIX}/software/`
* modules install path `${EASYBUILD_PREFIX}/modules/all`

### Easybuild main concepts

See also the [official Easybuild Tutorial: "Maintaining a Modern Scientific Software Stack Made Easy with EasyBuild"](https://easybuilders.github.io/easybuild-tutorial/2021-isc21/)

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

Easybuild is provided to you as a software module.

``` bash
module load tools/EasyBuild
```

In case you can't install the latest version yourself, please follow [the official instructions](http://easybuild.readthedocs.io/en/latest/Installation.html).
Nonetheless, we strongly recommand to use the provided module. 
**Don't forget** to setup your local Easybuild configuration first.

What is important for the installation of Easybuild are the following variables:

* `EASYBUILD_PREFIX`: where to install **local** modules and software, _i.e._ `$HOME/.local/easybuild`
* `EASYBUILD_MODULES_TOOL`: the type of [modules](http://modules.sourceforge.net/) tool you are using, _i.e._ `LMod` in this case
* `EASYBUILD_MODULE_NAMING_SCHEME`: the way the software and modules should be organized (flat view or hierarchical) -- we're advising on `CategorizedModuleNamingScheme`

**`/!\ IMPORTANT:`**  Recall that **you should be on a compute node to install [Easybuild](http://easybuild.readthedocs.io)** (otherwise the checks of the `module` command availability will fail.)

### Local Easybuild configuration

If you prefer to extend/complement the ULHPC software set while taking into account the cluster `$ULHPC_CLUSTER` ("iris" or "aion"), the toolchain version `<version>` (Ex: 2019b, 2020b etc.) and eventually the architecture `<arch>`.
In that case, you can use the following helper function defined at `/etc/profile.d/ulhpc_resif.sh`:

```bash
    resif-load-home-swset-prod
```
The function sets all the Lmod and Easybuild variables to match the production software set:

```bash
resif-load-home-swset-prod 
=> Set EASYBUILD_PREFIX to '/home/users/ekieffer/.local/easybuild/aion/2020b/epyc'
=> Enabling local RESIF home environment  (under /home/users/ekieffer) against '2020b' software set
   Current MODULEPATH=/home/users/ekieffer/.local/easybuild/aion/2020b/epyc/modules/all:/opt/apps/resif/aion/2020b/epyc/modules/all:/opt/apps/smc/modules/
```


For a shared project directory `<name>` located under `$PROJECTHOME/<name>`, you can use the following following helper scripts:
```bash
resif-load-project-swset-prod $PROJECTHOME/<name>
```
This function is very helpfull if you wish to share your custom software set with members of your group:

```bash
resif-load-project-swset-prod $PROJECTHOME/ulhpc-tutorials/
=> Set EASYBUILD_PREFIX to '/work/projects//ulhpc-tutorials//easybuild/aion/2020b/epyc'
=> Enabling local RESIF project environment  (under /work/projects//ulhpc-tutorials/) against '2020b' software set
   Current MODULEPATH=/work/projects//ulhpc-tutorials//easybuild/aion/2020b/epyc/modules/all:/opt/apps/resif/aion/2020b/epyc/modules/all:/opt/apps/smc/modules/
```



Additonnaly the ULHPC provides multiple functions to help you switching between software sets.

```bash
ekieffer@aion-0262(723109 -> 1:14:37) resif-<TAB>
resif-info                       resif-load-home-swset-legacy     resif-load-project-swset-devel   resif-load-project-swset-prod    resif-load-swset-legacy          resif-reset-swset
resif-load-home-swset-devel      resif-load-home-swset-prod       resif-load-project-swset-legacy  resif-load-swset-devel           resif-load-swset-prod            
```

### Install a missing software by complementing the ULHPC software set

Let's try to install the missing software

``` bash
(node)$ module spider BCFtools   # Complementaty tool to SAMTools
Lmod has detected the following error:  Unable to find: "BCFtools".

# Use helper function to setup local easybuild configuration
(node)$ resif-load-home-swset-prod 
# Load Easybuild
(node)$ module load tools/EasyBuild
# Search for recipes for the missing software
(node)$ eb -S BCFtools
== found valid index for /opt/apps/resif/aion/2020b/epyc/software/EasyBuild/4.5.4/easybuild/easyconfigs, so using it...
CFGS1=/opt/apps/resif/aion/2020b/epyc/software/EasyBuild/4.5.4/easybuild/easyconfigs
 * $CFGS1/b/BCFtools/BCFtools-1.2_extHTSlib_Makefile.patch
 * $CFGS1/b/BCFtools/BCFtools-1.3-foss-2016a.eb
 * $CFGS1/b/BCFtools/BCFtools-1.3-intel-2016a.eb
 * $CFGS1/b/BCFtools/BCFtools-1.3.1-foss-2016b.eb
 * $CFGS1/b/BCFtools/BCFtools-1.3_extHTSlib_Makefile.patch
 * $CFGS1/b/BCFtools/BCFtools-1.6-foss-2016b.eb
 * $CFGS1/b/BCFtools/BCFtools-1.6-foss-2017b.eb
 * $CFGS1/b/BCFtools/BCFtools-1.6-intel-2017b.eb
 * $CFGS1/b/BCFtools/BCFtools-1.8-GCC-6.4.0-2.28.eb
 * $CFGS1/b/BCFtools/BCFtools-1.9-foss-2018a.eb
 * $CFGS1/b/BCFtools/BCFtools-1.9-foss-2018b.eb
 * $CFGS1/b/BCFtools/BCFtools-1.9-iccifort-2019.1.144-GCC-8.2.0-2.31.1.eb
 * $CFGS1/b/BCFtools/BCFtools-1.9-intel-2018b.eb
 * $CFGS1/b/BCFtools/BCFtools-1.10.2-GCC-8.3.0.eb
 * $CFGS1/b/BCFtools/BCFtools-1.10.2-GCC-9.3.0.eb
 * $CFGS1/b/BCFtools/BCFtools-1.10.2-iccifort-2019.5.281.eb
 * $CFGS1/b/BCFtools/BCFtools-1.11-GCC-10.2.0.eb
 * $CFGS1/b/BCFtools/BCFtools-1.12-GCC-9.3.0.eb
 * $CFGS1/b/BCFtools/BCFtools-1.12-GCC-10.2.0.eb
 * $CFGS1/b/BCFtools/BCFtools-1.12-GCC-10.3.0.eb
 * $CFGS1/b/BCFtools/BCFtools-1.14-GCC-11.2.0.eb
```

From this list, you should select the version matching the target toolchain version -- see [ULHPC Toolchains and Software Set Versioning documentation](https://hpc-docs.uni.lu/environment/modules/#ulhpc-toolchains-and-software-set-versioning)
In particular, for 2020b version, the `GCC[core]` is set to 10.2.0 so `BCFtools-1.12-GCC-10.2.0.eb` seems like a promising candidate .

Once you pick a given recipy (for instance `BCFtools-1.12-GCC-10.2.0.eb`), install it with

       eb <name>.eb [-D] -r

* `-D` enables the dry-run mode to check what's going to be install -- **ALWAYS try it first**
* `-r` enables the robot mode to automatically install all dependencies while searching for easyconfigs in a set of pre-defined directories -- you can also prepend new directories to search for eb files (like the current directory `$PWD`) using the option and syntax `--robot-paths=$PWD:` (do not forget the ':'). See [Controlling the robot search path documentation](http://easybuild.readthedocs.io/en/latest/Using_the_EasyBuild_command_line.html#controlling-the-robot-search-path)
* The `$CFGS<n>/` prefix should be dropped unless you know what you're doing (and thus have previously defined the variable -- see the first output of the `eb -S [...]` command).

Let's try to review the missing dependencies from a dry-run :

``` bash
# Select the one matching the target software set version
(node)$ eb BCFtools-1.12-GCC-10.2.0.eb -Dr   # Dry-run
== Temporary log file in case of crash /tmp/eb-xvjew6tq/easybuild-45mth_zy.log
== found valid index for /opt/apps/resif/aion/2020b/epyc/software/EasyBuild/4.5.4/easybuild/easyconfigs, so using it...
== found valid index for /opt/apps/resif/aion/2020b/epyc/software/EasyBuild/4.5.4/easybuild/easyconfigs, so using it...
Dry run: printing build status of easyconfigs and dependencies
CFGS=/opt/apps/resif/aion/2020b/epyc/software/EasyBuild/4.5.4/easybuild/easyconfigs
 * [ ] $CFGS/m/M4/M4-1.4.19.eb (module: devel/M4/1.4.19)
 * [ ] $CFGS/b/Bison/Bison-3.8.2.eb (module: lang/Bison/3.8.2)
 * [x] $CFGS/f/flex/flex-2.6.4.eb (module: lang/flex/2.6.4)
 * [x] $CFGS/z/zlib/zlib-1.2.11.eb (module: lib/zlib/1.2.11)
 * [x] $CFGS/b/binutils/binutils-2.35.eb (module: tools/binutils/2.35)
 * [x] $CFGS/g/GCCcore/GCCcore-10.2.0.eb (module: compiler/GCCcore/10.2.0)
 * [x] $CFGS/z/zlib/zlib-1.2.11-GCCcore-10.2.0.eb (module: lib/zlib/1.2.11-GCCcore-10.2.0)
 * [x] $CFGS/n/ncurses/ncurses-6.2.eb (module: devel/ncurses/6.2)
 * [x] $CFGS/g/gettext/gettext-0.21.eb (module: tools/gettext/0.21)
 * [x] $CFGS/h/help2man/help2man-1.47.16-GCCcore-10.2.0.eb (module: tools/help2man/1.47.16-GCCcore-10.2.0)
 * [x] $CFGS/m/M4/M4-1.4.18-GCCcore-10.2.0.eb (module: devel/M4/1.4.18-GCCcore-10.2.0)
 * [x] $CFGS/b/Bison/Bison-3.7.1-GCCcore-10.2.0.eb (module: lang/Bison/3.7.1-GCCcore-10.2.0)
 * [x] $CFGS/f/flex/flex-2.6.4-GCCcore-10.2.0.eb (module: lang/flex/2.6.4-GCCcore-10.2.0)
 * [x] $CFGS/b/binutils/binutils-2.35-GCCcore-10.2.0.eb (module: tools/binutils/2.35-GCCcore-10.2.0)
 * [x] $CFGS/c/cURL/cURL-7.72.0-GCCcore-10.2.0.eb (module: tools/cURL/7.72.0-GCCcore-10.2.0)
 * [x] $CFGS/g/GCC/GCC-10.2.0.eb (module: compiler/GCC/10.2.0)
 * [x] $CFGS/b/bzip2/bzip2-1.0.8-GCCcore-10.2.0.eb (module: tools/bzip2/1.0.8-GCCcore-10.2.0)
 * [x] $CFGS/x/XZ/XZ-5.2.5-GCCcore-10.2.0.eb (module: tools/XZ/5.2.5-GCCcore-10.2.0)
 * [x] $CFGS/g/GSL/GSL-2.6-GCC-10.2.0.eb (module: numlib/GSL/2.6-GCC-10.2.0)
 * [x] $CFGS/h/HTSlib/HTSlib-1.12-GCC-10.2.0.eb (module: bio/HTSlib/1.12-GCC-10.2.0)
 * [ ] $CFGS/b/BCFtools/BCFtools-1.12-GCC-10.2.0.eb (module: bio/BCFtools/1.12-GCC-10.2.0)
== Temporary log file(s) /tmp/eb-xvjew6tq/easybuild-45mth_zy.log* have been removed.
== Temporary directory /tmp/eb-xvjew6tq has been removed.
```
Let's try to install it (remove the `-D`):

``` bash
# Select the one matching the target software set version
(node)$ eb BCFtools-1.12-GCC-10.2.0.eb -r
```
From now on, you should be able to see the new module.

```bash
(node)$  module spider BCF

-----------------------------------------------------------------------------------------------------
  bio/BCFtools: bio/BCFtools/1.12-GCC-10.2.0
-----------------------------------------------------------------------------------------------------
    Description:
      Samtools is a suite of programs for interacting with high-throughput sequencing data. BCFtools -
      Reading/writing BCF2/VCF/gVCF files and calling/filtering/summarising SNP and short indel sequence
      variants
   This module can be loaded directly: module load bio/BCFtools/1.12-GCC-10.2.0

    Help:

      Description
      ===========
      Samtools is a suite of programs for interacting with high-throughput sequencing data.
       BCFtools - Reading/writing BCF2/VCF/gVCF files and calling/filtering/summarising SNP and short indel sequence
       variants


      More information
      ================
       - Homepage: https://www.htslib.org/
```


**Tips**: When you load a module `<NAME>` generated by Easybuild, it is installed within the directory reported by the `$EBROOT<NAME>` variable.
In the above case, you will find the generated binary in `${EBROOTBCFTOOLS}/`.

### Install a missing software with a more recent toolchain

The release of a new software set takes some time and you may wish to use [the last toolchain]( https://docs.easybuild.io/common-toolchains/#common_toolchains_update_cycle) provided by Easybuild.  

Let's say we wish to install BCFtools with GCC-12.2.0 (which is part of foss-2022b). 

First, you will need to add the following content to your `$HOME/.bashrc`. We recommend you to add it as a bash function to be able to switch between Easyconfig environement.

``` bash
load_local_easybuild(){
# EASYBUILD_PREFIX: [basedir]/<cluster>/<environment>/<arch>
# Ex: Default EASYBUILD_PREFIX in your home - Adapt to project directory if needed
_VERSION="${1:-${RESIF_VERSION_PROD}}"
_EB_PREFIX=$HOME/.local/easybuild
# ... eventually complemented with cluster
[ -n "${ULHPC_CLUSTER}" ] && _EB_PREFIX="${_EB_PREFIX}/${ULHPC_CLUSTER}"
# ... eventually complemented with software set version
_EB_PREFIX="${_EB_PREFIX}/${_VERSION}"
# ... eventually complemented with arch
[ -n "${RESIF_ARCH}" ] && _EB_PREFIX="${_EB_PREFIX}/${RESIF_ARCH}"
export EASYBUILD_PREFIX="${_EB_PREFIX}"
export LOCAL_MODULES=${EASYBUILD_PREFIX}/modules/all
}
```

Suppose also that we need the last easybuild version, i.e., v4.7.1.
In order to avoid your local version to collide with the module one, we suggest you to install the newest easybuild inside a virtualenv as follows:

```bash
# Purge all loaded modules
module purge
# use the system python and install Easybuild
export EB_PYTHON=$(which python3)
# load local easybuild version
load_local_easybuild "2022b"
# double-check the EASYBUILD_PREFIX
echo ${EASYBUILD_PREFIX}
# install easybuild
python3 -m pip install easybuild==4.7.1 --user
# double-check the installed version
eb --version
```
Now, let's try to install BCFtools with GCC-12.2.0. Using the command `eb -S BCFtools`, we can check the existing versions as previsouly

```bash
# Search for recipes for the missing software
(node)$ eb -S BCFtools
== found valid index for /home/users/ekieffer/.local/easybuild/easyconfigs, so using it...
CFGS1=/home/users/ekieffer/.local/easybuild/easyconfigs
 * $CFGS1/b/BCFtools/BCFtools-1.2_extHTSlib_Makefile.patch
 * $CFGS1/b/BCFtools/BCFtools-1.3-foss-2016a.eb
 * $CFGS1/b/BCFtools/BCFtools-1.3-intel-2016a.eb
 * $CFGS1/b/BCFtools/BCFtools-1.3.1-foss-2016b.eb
 * $CFGS1/b/BCFtools/BCFtools-1.3_extHTSlib_Makefile.patch
 * $CFGS1/b/BCFtools/BCFtools-1.6-foss-2016b.eb
 * $CFGS1/b/BCFtools/BCFtools-1.6-foss-2017b.eb
 * $CFGS1/b/BCFtools/BCFtools-1.6-intel-2017b.eb
 * $CFGS1/b/BCFtools/BCFtools-1.8-GCC-6.4.0-2.28.eb
 * $CFGS1/b/BCFtools/BCFtools-1.9-foss-2018a.eb
 * $CFGS1/b/BCFtools/BCFtools-1.9-foss-2018b.eb
 * $CFGS1/b/BCFtools/BCFtools-1.9-iccifort-2019.1.144-GCC-8.2.0-2.31.1.eb
 * $CFGS1/b/BCFtools/BCFtools-1.9-intel-2018b.eb
 * $CFGS1/b/BCFtools/BCFtools-1.10.2-GCC-8.3.0.eb
 * $CFGS1/b/BCFtools/BCFtools-1.10.2-GCC-9.3.0.eb
 * $CFGS1/b/BCFtools/BCFtools-1.10.2-iccifort-2019.5.281.eb
 * $CFGS1/b/BCFtools/BCFtools-1.11-GCC-10.2.0.eb
 * $CFGS1/b/BCFtools/BCFtools-1.12-GCC-9.3.0.eb
 * $CFGS1/b/BCFtools/BCFtools-1.12-GCC-10.2.0.eb
 * $CFGS1/b/BCFtools/BCFtools-1.12-GCC-10.3.0.eb
 * $CFGS1/b/BCFtools/BCFtools-1.14-GCC-11.2.0.eb
 * $CFGS1/b/BCFtools/BCFtools-1.15.1-GCC-11.3.0.eb
 * $CFGS1/b/BCFtools/BCFtools-1.17-GCC-12.2.0.eb

Note: 4 matching archived easyconfig(s) found, use --consider-archived-easyconfigs to see them
```

We are going to install `BCFtools-1.17-GCC-12.2.0.eb` but let's try with the dry-run option first. We should see that all dependencies are not satisfied.

```bash
eb -D BCFtools-1.17-GCC-12.2.0.eb -r
== Temporary log file in case of crash /tmp/eb-xasu197x/easybuild-up2jaclx.log
== found valid index for /home/users/ekieffer/.local/easybuild/easyconfigs, so using it...
== found valid index for /home/users/ekieffer/.local/easybuild/easyconfigs, so using it...
Dry run: printing build status of easyconfigs and dependencies
CFGS=/home/users/ekieffer/.local/easybuild/easyconfigs
 * [ ] $CFGS/m/M4/M4-1.4.19.eb (module: devel/M4/1.4.19)
 * [ ] $CFGS/b/Bison/Bison-3.8.2.eb (module: lang/Bison/3.8.2)
 * [ ] $CFGS/f/flex/flex-2.6.4.eb (module: lang/flex/2.6.4)
 * [ ] $CFGS/z/zlib/zlib-1.2.12.eb (module: lib/zlib/1.2.12)
 * [ ] $CFGS/b/binutils/binutils-2.39.eb (module: tools/binutils/2.39)
 * [ ] $CFGS/g/GCCcore/GCCcore-12.2.0.eb (module: compiler/GCCcore/12.2.0)
 * [ ] $CFGS/z/zlib/zlib-1.2.12-GCCcore-12.2.0.eb (module: lib/zlib/1.2.12-GCCcore-12.2.0)
 * [ ] $CFGS/n/ncurses/ncurses-6.3.eb (module: devel/ncurses/6.3)
 * [ ] $CFGS/g/gettext/gettext-0.21.1.eb (module: tools/gettext/0.21.1)
 * [ ] $CFGS/h/help2man/help2man-1.49.2-GCCcore-12.2.0.eb (module: tools/help2man/1.49.2-GCCcore-12.2.0)
 * [ ] $CFGS/m/M4/M4-1.4.19-GCCcore-12.2.0.eb (module: devel/M4/1.4.19-GCCcore-12.2.0)
 * [ ] $CFGS/p/pkgconf/pkgconf-1.8.0.eb (module: devel/pkgconf/1.8.0)
 * [ ] $CFGS/b/Bison/Bison-3.8.2-GCCcore-12.2.0.eb (module: lang/Bison/3.8.2-GCCcore-12.2.0)
 * [ ] $CFGS/o/OpenSSL/OpenSSL-1.1.eb (module: system/OpenSSL/1.1)
 * [ ] $CFGS/f/flex/flex-2.6.4-GCCcore-12.2.0.eb (module: lang/flex/2.6.4-GCCcore-12.2.0)
 * [ ] $CFGS/b/binutils/binutils-2.39-GCCcore-12.2.0.eb (module: tools/binutils/2.39-GCCcore-12.2.0)
 * [ ] $CFGS/c/cURL/cURL-7.86.0-GCCcore-12.2.0.eb (module: tools/cURL/7.86.0-GCCcore-12.2.0)
 * [ ] $CFGS/g/GCC/GCC-12.2.0.eb (module: compiler/GCC/12.2.0)
 * [ ] $CFGS/b/bzip2/bzip2-1.0.8-GCCcore-12.2.0.eb (module: tools/bzip2/1.0.8-GCCcore-12.2.0)
 * [ ] $CFGS/x/XZ/XZ-5.2.7-GCCcore-12.2.0.eb (module: tools/XZ/5.2.7-GCCcore-12.2.0)
 * [ ] $CFGS/g/GSL/GSL-2.7-GCC-12.2.0.eb (module: numlib/GSL/2.7-GCC-12.2.0)
 * [ ] $CFGS/h/HTSlib/HTSlib-1.17-GCC-12.2.0.eb (module: bio/HTSlib/1.17-GCC-12.2.0)
 * [ ] $CFGS/b/BCFtools/BCFtools-1.17-GCC-12.2.0.eb (module: bio/BCFtools/1.17-GCC-12.2.0)
== Temporary log file(s) /tmp/eb-xasu197x/easybuild-up2jaclx.log* have been removed.
== Temporary directory /tmp/eb-xasu197x has been removed.
```


Now, let's install it with `eb BCFtools-1.17-GCC-12.2.0.eb -r`. It takes a bit of time to complete since all dependencies need to be installed first.


### Build software using a customized EasyConfig file

There are multiple ways to amend an EasyConfig file. Check the `--try-*` option flags for all the possibilities.

Generally you want to do that when the up-to-date version of the software you want is **not** available as a recipy within Easybuild.
For instance, let's consider the most recent version, i.e., 20230422, of the [GNU parallel](https://www.gnu.org/software/parallel/).

It is not available as module, so let's build it.

First let's check for available easyconfigs recipies if one exist for the expected version:

```
(node)$> eb -S parallel
== found valid index for /home/users/ekieffer/.local/easybuild/easyconfigs, so using it...
CFGS1=/home/users/ekieffer/.local/easybuild/easyconfigs
[...]
 * $CFGS1/p/parallel/parallel-20141122-GCC-4.9.2.eb
 * $CFGS1/p/parallel/parallel-20150322-GCC-4.9.2.eb
 * $CFGS1/p/parallel/parallel-20150822-GCC-4.9.2.eb
 * $CFGS1/p/parallel/parallel-20160622-foss-2016a.eb
 * $CFGS1/p/parallel/parallel-20170822-intel-2017a.eb
 * $CFGS1/p/parallel/parallel-20171022-intel-2017b.eb
 * $CFGS1/p/parallel/parallel-20171122-foss-2017b.eb
 * $CFGS1/p/parallel/parallel-20171122-intel-2017b.eb
 * $CFGS1/p/parallel/parallel-20180422-intel-2018a.eb
 * $CFGS1/p/parallel/parallel-20180822-foss-2018b.eb
 * $CFGS1/p/parallel/parallel-20181222-intel-2018b.eb
 * $CFGS1/p/parallel/parallel-20190222-GCCcore-7.3.0.eb
 * $CFGS1/p/parallel/parallel-20190622-GCCcore-8.2.0.eb
 * $CFGS1/p/parallel/parallel-20190922-GCCcore-8.3.0.eb
 * $CFGS1/p/parallel/parallel-20200422-GCCcore-9.3.0.eb
 * $CFGS1/p/parallel/parallel-20200522-GCCcore-9.3.0.eb
 * $CFGS1/p/parallel/parallel-20210322-GCCcore-10.2.0.eb
 * $CFGS1/p/parallel/parallel-20210622-GCCcore-10.3.0.eb
 * $CFGS1/p/parallel/parallel-20210722-GCCcore-11.2.0.eb
 * $CFGS1/p/parallel/parallel-20220722-GCCcore-11.3.0.eb
 * $CFGS1/r/R/DMCfun-1.3.0_fix-parallel-detect.patch
 * $CFGS1/w/WRF/WRF_parallel_build_fix.patch
 * $CFGS1/x/Xmipp/Xmipp-3.19.04-Apollo_add_missing_pthread_to_XmippParallel.patch

Note: 7 matching archived easyconfig(s) found, use --consider-archived-easyconfigs to see them
```
As you can see, parallel-20220722-GCCcore-11.3.0.eb is a good candidate to start with. It is the most recent easybuild existing in the repository.

We are going to reuse one of the latest EasyConfig available, for instance lets copy `$CFGS1/p/parallel/parallel-20220722-GCCcore-11.3.0.eb` as it was the most recent.
We'll have to make it match the toolchain/compiler available defined previously i.e. GCCcore-12.2.0.

```bash
# Work in a dedicated directory
(node)$> mkdir -p ~/software/parallel
(node)$> cd ~/software/parallel

# collect the definition of the CFGS1 variable
(node)$> CFGS1=/home/users/ekieffer/.local/easybuild/easyconfigs
(node)$> cp $CFGS1/p/parallel/parallel-20220722-GCCcore-11.3.0.eb .
# Adapt the filename with the target version and your default building environement 
(node)$> mv parallel-20220722-GCCcore-11.3.0.eb parallel-20230422-GCCcore-12.2.0.eb  
```

Please open `parallel-20230422-GCCcore-12.2.0.eb` with your preferred editor (e.g., vim, nano, emacs). You should see the following:

```python
easyblock = 'ConfigureMake'

name = 'parallel'
version = '20220722'

homepage = 'https://savannah.gnu.org/projects/parallel/'
description = """parallel: Build and execute shell commands in parallel"""

toolchain = {'name': 'GCCcore', 'version': '11.3.0'}

source_urls = [GNU_SOURCE]
sources = [SOURCELOWER_TAR_BZ2]
checksums = ['0e4083ac0d850c434598c6dfbf98f3b6dd2cc932a3af9269eb1f9323e43af019']

builddependencies = [('binutils', '2.38')]

dependencies = [('Perl', '5.34.1')]

sanity_check_paths = {
    'files': ['bin/parallel'],
    'dirs': []
}

sanity_check_commands = ["parallel --help"]

moduleclass = 'tools'
```
We will need to update the following fields:

* version
* toolchain
* builddependencies
* dependencies

Let's first update the (build)dependencies to match the new GCCcore-12.2.0.

To find the appropriate version for the dependencies, we need to search the available versions for binutils and Perl regarding GCCcore-12.2.0.

``` bash
# Searching binutils version
eb -S binutils
[...]
* $CFGS1/b/binutils/binutils-2.39-GCCcore-12.2.0.eb
[...]
# Searching Perl version
eb -S Perl
[...]
* $CFGS1/p/Perl/Perl-5.36.0-GCCcore-12.2.0.eb
[...]
```

Once the version of all dependencies have been found, we are now able to update `parallel-20230422-GCCcore-12.2.0.eb`.
Before diplaying the last diff between the old easybuild and the new one, we also need to update the checksum. For this purpose, Easybuild implements a nice functionnality which injects directly the right checksum into the eb file.

```bash
# Injecting the checksum
eb --inject-checksums='sha256'  --force parallel-20230422-GCCcore-12.2.0.eb
== Temporary log file in case of crash /tmp/eb-ed9gqepv/easybuild-c1u3btwk.log
== found valid index for /home/users/ekieffer/.local/easybuild/easyconfigs, so using it...
== found valid index for /home/users/ekieffer/.local/easybuild/easyconfigs, so using it...
== injecting sha256 checksums in /mnt/irisgpfs/users/ekieffer/software/parallel/parallel-20230422-GCCcore-12.2.0.eb
== fetching sources & patches for parallel-20230422-GCCcore-12.2.0.eb...

WARNING: Found existing checksums in parallel-20230422-GCCcore-12.2.0.eb, overwriting them (due to use of --force)...

== backup of easyconfig file saved to /mnt/irisgpfs/users/ekieffer/software/parallel/parallel-20230422-GCCcore-12.2.0.eb.bak_20230503200710_3555416...
== injecting sha256 checksums for sources & patches in parallel-20230422-GCCcore-12.2.0.eb...
== * parallel-20230422.tar.bz2: 9106593d09dc4de0e094b7b14390a309d8fcb1d27104a53814d16937dcbae3c2
== Temporary log file(s) /tmp/eb-ed9gqepv/easybuild-c1u3btwk.log* have been removed.
== Temporary directory /tmp/eb-ed9gqepv has been removed.
```

Before updating the checksums inside the eb file, Easybuild creates a backup. We are now able to display a diff between the original and the new eb files with the following command `diff -u --color parallel-20220722-GCCcore-11.3.0.eb parallel-20230422-GCCcore-12.2.0.eb`.

* You can also find the new easybuild recipy in the `.../tutorials/tools/easybuild/` folder.

```diff
--- parallel-20220722-GCCcore-11.3.0.eb 2023-05-03 20:01:58.538044000 +0200
+++ parallel-20230422-GCCcore-12.2.0.eb 2023-05-03 20:07:10.034852000 +0200
@@ -1,20 +1,20 @@
 easyblock = 'ConfigureMake'

 name = 'parallel'
-version = '20220722'
+version = '20230422'

 homepage = 'https://savannah.gnu.org/projects/parallel/'
 description = """parallel: Build and execute shell commands in parallel"""

-toolchain = {'name': 'GCCcore', 'version': '11.3.0'}
+toolchain = {'name': 'GCCcore', 'version': '12.2.0'}

 source_urls = [GNU_SOURCE]
 sources = [SOURCELOWER_TAR_BZ2]
-checksums = ['0e4083ac0d850c434598c6dfbf98f3b6dd2cc932a3af9269eb1f9323e43af019']
+checksums = ['9106593d09dc4de0e094b7b14390a309d8fcb1d27104a53814d16937dcbae3c2']

-builddependencies = [('binutils', '2.38')]
+builddependencies = [('binutils', '2.39')]

-dependencies = [('Perl', '5.34.1')]
+dependencies = [('Perl', '5.36.0')]

 sanity_check_paths = {
     'files': ['bin/parallel'],

```

You can now build it

```bash
(node)$> eb parallel-20230422-GCCcore-12.2.0.eb -Dr 
== Temporary log file in case of crash /tmp/eb-j2gcfbzy/easybuild-2uaaamz_.log
== found valid index for /home/users/ekieffer/.local/easybuild/easyconfigs, so using it...
== found valid index for /home/users/ekieffer/.local/easybuild/easyconfigs, so using it...
Dry run: printing build status of easyconfigs and dependencies
 * [x] /home/users/ekieffer/.local/easybuild/easyconfigs/m/M4/M4-1.4.19.eb (module: devel/M4/1.4.19)
 * [x] /home/users/ekieffer/.local/easybuild/easyconfigs/b/Bison/Bison-3.8.2.eb (module: lang/Bison/3.8.2)
 * [x] /home/users/ekieffer/.local/easybuild/easyconfigs/f/flex/flex-2.6.4.eb (module: lang/flex/2.6.4)
 * [x] /home/users/ekieffer/.local/easybuild/easyconfigs/z/zlib/zlib-1.2.12.eb (module: lib/zlib/1.2.12)
 * [x] /home/users/ekieffer/.local/easybuild/easyconfigs/b/binutils/binutils-2.39.eb (module: tools/binutils/2.39)
 * [x] /home/users/ekieffer/.local/easybuild/easyconfigs/g/GCCcore/GCCcore-12.2.0.eb (module: compiler/GCCcore/12.2.0)
 * [x] /home/users/ekieffer/.local/easybuild/easyconfigs/z/zlib/zlib-1.2.12-GCCcore-12.2.0.eb (module: lib/zlib/1.2.12-GCCcore-12.2.0)
 * [x] /home/users/ekieffer/.local/easybuild/easyconfigs/h/help2man/help2man-1.49.2-GCCcore-12.2.0.eb (module: tools/help2man/1.49.2-GCCcore-12.2.0)
 * [x] /home/users/ekieffer/.local/easybuild/easyconfigs/m/M4/M4-1.4.19-GCCcore-12.2.0.eb (module: devel/M4/1.4.19-GCCcore-12.2.0)
 * [x] /home/users/ekieffer/.local/easybuild/easyconfigs/b/Bison/Bison-3.8.2-GCCcore-12.2.0.eb (module: lang/Bison/3.8.2-GCCcore-12.2.0)
 * [x] /home/users/ekieffer/.local/easybuild/easyconfigs/f/flex/flex-2.6.4-GCCcore-12.2.0.eb (module: lang/flex/2.6.4-GCCcore-12.2.0)
 * [x] /home/users/ekieffer/.local/easybuild/easyconfigs/b/binutils/binutils-2.39-GCCcore-12.2.0.eb (module: tools/binutils/2.39-GCCcore-12.2.0)
 * [ ] /home/users/ekieffer/.local/easybuild/easyconfigs/g/groff/groff-1.22.4-GCCcore-12.2.0.eb (module: tools/groff/1.22.4-GCCcore-12.2.0)
 * [ ] /home/users/ekieffer/.local/easybuild/easyconfigs/e/expat/expat-2.4.9-GCCcore-12.2.0.eb (module: tools/expat/2.4.9-GCCcore-12.2.0)
 * [ ] /home/users/ekieffer/.local/easybuild/easyconfigs/n/ncurses/ncurses-6.3-GCCcore-12.2.0.eb (module: devel/ncurses/6.3-GCCcore-12.2.0)
 * [ ] /home/users/ekieffer/.local/easybuild/easyconfigs/l/libreadline/libreadline-8.2-GCCcore-12.2.0.eb (module: lib/libreadline/8.2-GCCcore-12.2.0)
 * [x] /home/users/ekieffer/.local/easybuild/easyconfigs/p/pkgconf/pkgconf-1.8.0.eb (module: devel/pkgconf/1.8.0)
 * [x] /home/users/ekieffer/.local/easybuild/easyconfigs/o/OpenSSL/OpenSSL-1.1.eb (module: system/OpenSSL/1.1)
 * [ ] /home/users/ekieffer/.local/easybuild/easyconfigs/d/DB/DB-18.1.40-GCCcore-12.2.0.eb (module: tools/DB/18.1.40-GCCcore-12.2.0)
 * [ ] /home/users/ekieffer/.local/easybuild/easyconfigs/p/Perl/Perl-5.36.0-GCCcore-12.2.0.eb (module: lang/Perl/5.36.0-GCCcore-12.2.0)
 * [ ] /mnt/irisgpfs/users/ekieffer/software/parallel/parallel-20230422-GCCcore-12.2.0.eb (module: tools/parallel/20230422-GCCcore-12.2.0)
== Temporary log file(s) /tmp/eb-j2gcfbzy/easybuild-2uaaamz_.log* have been removed.
== Temporary directory /tmp/eb-j2gcfbzy has been removed.
(node)$> eb parallel-20230422-GCCcore-12.2.0.eb -r
```

Check the result:

```bash
(node)$> module av parallel

------------------------------------------------------------------------------------ /home/users/ekieffer/.local/easybuild/aion/2022b/epyc/modules/all -------------------------------------------------------------------------------------
   tools/parallel/20230422-GCCcore-12.2.0

If the avail list is too long consider trying:

"module --default avail" or "ml -d av" to just list the default modules.
"module overview" or "ml ov" to display the number of modules for each name.

Use "module spider" to find all possible modules and extensions.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".


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

While modifying easyconfig recipes, the following commands can be handy:

- `eb --available-module-tools`: List all available tools for installing and managing modules.
- `eb --list-easyblocks`: List Python easyblock files available to execute an easybuild recipe. Select by setting the `easyblock` parameter in the recipe file. For example, setting `easyblock = CMakeMake` builds a recipe with Cmake+Make, and `easyblock = CMakeNinja` builds with CMake+Ninja.
- `eb --avail-easyconfig-params --easyblock <block name>`: List the setting available for recipe files using the given easyblock.

# Submitting working Easyconfigs to easybuilders

* Follow the __Official documentations__:
    - [Integration with Github](https://easybuild.readthedocs.io/en/latest/Integration_with_GitHub.html)
    - [Submitting pull requests (`--new-pr`)](https://easybuild.readthedocs.io/en/latest/Integration_with_GitHub.html#submitting-pull-requests-new-pr)
    - [Uploading test reports (`--upload-test-report`)](https://easybuild.readthedocs.io/en/latest/Integration_with_GitHub.html#uploading-test-reports-upload-test-report)
    - [Updating existing pull requests (`--update-pr`)](https://easybuild.readthedocs.io/en/master/Integration_with_GitHub.html#updating-existing-pull-requests-update-pr)


## To go further (to update)

- [ULHPC/sw](https://github.com/ULHPC/sw): RESIF 3 sources
- [EasyBuild homepage](http://easybuilders.github.io/easybuild)
- [EasyBuild documentation](http://easybuilders.github.io/easybuild/)
- [Getting started](https://github.com/easybuilders/easybuild/wiki/Getting-started)
- [Using EasyBuild](https://github.com/easybuilders/easybuild/wiki/Using-EasyBuild)
- [Step-by-step guide](https://github.com/easybuilders/easybuild/wiki/Step-by-step-guide)
