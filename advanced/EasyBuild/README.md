-*- mode: markdown;mode:visual-line;  fill-column: 80 -*-

Copyright (c) 2014-2017 UL HPC Team  -- see <http://hpc.uni.lu>

---------------------------------------------------------
# Building [custom] software with EasyBuild on UL HPC platform

[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/advanced/EasyBuild/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/advanced/EasyBuild/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/advanced/Easybuild/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

[![](https://github.com/ULHPC/tutorials/raw/devel/advanced/EasyBuild/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/advanced/EasyBuild/slides.pdf)


The objective of this tutorial is to show how [EasyBuild](https://github.com/hpcugent/easybuild) can be used to ease, automate and script the build of software on the UL HPC platforms.

Two use-cases are considered. First, we are going to build software that are supported by EasyBuild. In a second time, we will see through a simple example how to add support for a new software in EasyBuild.

The benefit of using EasyBuild for your builds is that it allows automated and reproducable build of software. Once a build has been made, the build script (via the *EasyConfig file*) or the installed software (via the *module file*) can be shared with other users.

Before starting this tutorial, ensure you are able to [connect to the chaos and gaia cluster](https://hpc.uni.lu/users/docs/access.html).
**For all your compilation with Easybuild, you must work on a computing node:**

* Gaia

		(access-gaia)$> oarsub -I -l core=1,walltime=4

* Iris

		(access-iris)$> srun -p interactive --qos qos-interactive -t 0-4:0:0 --pty bash

The latest version of this tutorial is available on
[Github](https://github.com/ULHPC/tutorials/tree/devel/advanced/EasyBuild).

## Short introduction to EasyBuild

EasyBuild is a tool that allows to perform automated and reproducible compilation and installation of software. A large number of scientific software are supported (649 software packages in the last release).

All builds and installations are performed at user level, so you don't need the admin rights.
The software are installed in your home directory (by default in `$HOME/.local/easybuild/software/`) and a module file is generated (by default in `$HOME/.local/easybuild/modules/`) to use the software.

EasyBuild relies on two main concepts: *Toolchains* and *EasyConfig file*.

A **toolchain** corresponds to a compiler and a set of libraries which are commonly used to build a software. The two main toolchains frequently used on the UL HPC platform are the GOOLF and the ICTCE toolchains. GOOLF is based on the GCC compiler and on open-source libraries (OpenMPI, OpenBLAS, etc.). ICTCE is based on the Intel compiler and on Intel libraries (Intel MPI, Intel Math Kernel Library, etc.).

An **EasyConfig file** is a simple text file that describes the build process of a software. For most software that uses standard procedure (like `configure`, `make` and `make install`), this file is very simple. Many EasyConfig files are already provided with EasyBuild.


By default, EasyConfig files and generated modules are named using the following convention:
`<Software-Name>-<Software-Version>-<Toolchain-Name>-<Toolchain-Version>`

On the cluster however, for the module names we use a custom naming convention that is explained in the RESIF tutorial:
`<Software-Class>/<Software-Name>/<Software-Version>-<Toolchain-Name>-<Toolchain-Version>`

Additional details are available on EasyBuild website:

- [EasyBuild homepage](http://hpcugent.github.io/easybuild)
- [EasyBuild documentation](http://hpcugent.github.io/easybuild/)
- [What is EasyBuild?](https://github.com/hpcugent/easybuild/wiki/EasyBuild)
- [Toolchains](https://github.com/hpcugent/easybuild/wiki/Compiler-toolchains)
- [EasyConfig files](https://github.com/hpcugent/easybuild/wiki/Easyconfig-files)
- [List of supported software packages](https://github.com/hpcugent/easybuild/wiki/List-of-supported-software-packages)

## Installing Easybuild

You probably want the latest version of Easybuild so we are going here to install it following [the official instructions](http://easybuild.readthedocs.io/en/latest/Installation.html).

Add the following entries to your `~/.bashrc`:

```bash
export EASYBUILD_PREFIX=$HOME/.local/easybuild
export EASYBUILD_MODULES_TOOL=Lmod
export EASYBUILD_MODULE_NAMING_SCHEME=CategorizedModuleNamingScheme
# Use the below variable to run:
#    module use $LOCAL_MODULES
#    module load tools/EasyBuild
export LOCAL_MODULES=${EASYBUILD_PREFIX}/modules/all
```

Then source this file to expose the environment variables:

```bash
$> source ~/.bashrc
$> echo $EASYBUILD_PREFIX
/home/users/svarrette/.local/easybuild
```

Now let's install Easybuild following the [boostrapping procedure](http://easybuild.readthedocs.io/en/latest/Installation.html#bootstrapping-easybuild)

```bash
$> cd /tmp/
# download script
curl -o /tmp/bootstrap_eb.py  https://raw.githubusercontent.com/hpcugent/easybuild-framework/develop/easybuild/scripts/bootstrap_eb.py

# install Easybuild
$> python /tmp/bootstrap_eb.py $EASYBUILD_PREFIX

# Load it
$> echo $MODULEPATH
$> module use $LOCAL_MODULES
$> echo $MODULEPATH
$> module spider Easybuild
$> module load tools/EasyBuild
```





## EasyBuild on UL HPC platform

To use EasyBuild on a compute node, load the EasyBuild module (if available):


    $> module avail EasyBuild

    ------------- /opt/apps/resif/devel/v1.1-20150414/core/modules/tools -------------
        tools/EasyBuild/2.0.0

    ------------- /opt/apps/resif/devel/v1.1-20150414/core/modules/base -------------
        base/EasyBuild/install-2.1.0

    $> module load base/EasyBuild/install-2.1.0

You can also install EasyBuild yourself with the `bootstrap_eb.py` script provided by EasyBuild:

	$> wget https://raw.githubusercontent.com/hpcugent/easybuild-framework/develop/easybuild/scripts/bootstrap_eb.py
	$> EASYBUILD_MODULES_TOOL=Lmod EASYBUILD_MODULE_NAMING_SCHEME=CategorizedModuleNamingScheme python bootstrap_eb.py $HOME/.local/easybuild
	$> module use $HOME/.local/easybuild/modules/all
	$> module load tools/EasyBuild
	$> echo "export EASYBUILD_MODULE_NAMING_SCHEME=CategorizedModuleNamingScheme" >> ~/.bashrc
	$> source ~/.bashrc

The EasyBuild command is `eb`. Check the version you have loaded:

    $> eb --version

    This is EasyBuild 2.1.0dev-r6fee583a88e99d1384314790a419c83e85f18f3d (framework: 2.1.0dev-r2aa673bb5f61cb2d65e4a3037cc2337e6df2d3e6, easyblocks: 2.1.0dev-r6fee583a88e99d1384314790a419c83e85f18f3d) on host h-cluster1-11.


Note that this version number from the modules on Gaia and Chaos are a bit peculiar because this is a custom installation on the cluster.

To get help on the EasyBuild options, use the `-h` or `-H` option flags:

    $> eb -h
    $> eb -H




## Build software using provided EasyConfig file

In this part, we propose to build High Performance Linpack (HPL) using EasyBuild.
HPL is supported by EasyBuild, this means that an EasyConfig file allowing to build HPL is already provided with EasyBuild.

### Gaia
First, let's see which HPL are available on the cluster:

    $> module avail HPL

    ------------- /opt/apps/resif/devel/v1.1-20150414/core/modules/tools -------------
        tools/HPL/2.0-goolf-1.4.10


Then, search for available EasyConfig files with HPL in their name. The EasyConfig files are named with the `.eb` extension.

    $> eb -S HPL

    == temporary log file in case of crash /tmp/eb-p2DT7H/easybuild-ligIot.log
    == Searching (case-insensitive) for 'HPL' in /opt/apps/resif/devel/v1.1-20150414/.installRef/easybuild-easyconfigs/easybuild/easyconfigs
    CFGS1=/opt/apps/resif/devel/v1.1-20150414/.installRef/easybuild-easyconfigs/easybuild/easyconfigs/h/HPL
     * $CFGS1/HPL-2.0-cgmpolf-1.1.6.eb
     * $CFGS1/HPL-2.0-cgmvolf-1.1.12rc1.eb
     * $CFGS1/HPL-2.0-cgmvolf-1.2.7.eb
     * $CFGS1/HPL-2.0-cgoolf-1.1.7.eb
     * $CFGS1/HPL-2.0-foss-2014b.eb
     * $CFGS1/HPL-2.0-goalf-1.1.0-no-OFED.eb
     * $CFGS1/HPL-2.0-goolf-1.4.10.eb
     * $CFGS1/HPL-2.0-goolf-1.5.16.eb
     * $CFGS1/HPL-2.0-ictce-4.0.6.eb
     * $CFGS1/HPL-2.0-ictce-5.3.0.eb
     * $CFGS1/HPL-2.0-ictce-6.0.5.eb
     * $CFGS1/HPL-2.0-ictce-6.1.5.eb
     * $CFGS1/HPL-2.0-iomkl-4.6.13.eb
     * $CFGS1/HPL-2.1-foss-2015a.eb
     * $CFGS1/HPL-2.1-gimkl-1.5.9.eb
     * $CFGS1/HPL-2.1-gmpolf-1.4.8.eb
     * $CFGS1/HPL-2.1-gmvolf-1.7.20.eb
     * $CFGS1/HPL-2.1-goolf-1.7.20.eb
     * $CFGS1/HPL-2.1-goolfc-1.4.10.eb
     * $CFGS1/HPL-2.1-goolfc-2.6.10.eb
     * $CFGS1/HPL-2.1-gpsolf-2014.12.eb
     * $CFGS1/HPL-2.1-ictce-6.3.5.eb
     * $CFGS1/HPL-2.1-ictce-7.1.2.eb
     * $CFGS1/HPL-2.1-intel-2014.10.eb
     * $CFGS1/HPL-2.1-intel-2014.11.eb
     * $CFGS1/HPL-2.1-intel-2014b.eb
     * $CFGS1/HPL-2.1-intel-2015.02.eb
     * $CFGS1/HPL-2.1-intel-2015a.eb
     * $CFGS1/HPL-2.1-intel-para-2014.12.eb
     * $CFGS1/HPL-2.1-iomkl-2015.01.eb
     * $CFGS1/HPL-2.1-iomkl-2015.02.eb
     * $CFGS1/HPL_parallel-make.patch
    == temporary log file(s) /tmp/eb-p2DT7H/easybuild-ligIot.log* have been removed.
    == temporary directory /tmp/eb-p2DT7H has been removed.


If we try to build `HPL-2.0-goolf-1.4.10`, nothing will be done as it is already installed on the cluster.

    $> eb HPL-2.0-goolf-1.4.10.eb

    == temporary log file in case of crash /tmp/eb-JKadCH/easybuild-SoXdix.log
    == tools/HPL/2.0-goolf-1.4.10 is already installed (module found), skipping
    == No easyconfigs left to be built.
    == Build succeeded for 0 out of 0
    == temporary log file(s) /tmp/eb-JKadCH/easybuild-SoXdix.log* have been removed.
    == temporary directory /tmp/eb-JKadCH has been removed.


However the build can be forced using the `-f` option flag. Then this software will be re-built.
(Tip: prefix your command with `time` to know its duration)

    $> time eb HPL-2.0-goolf-1.4.10.eb -f

    == temporary log file in case of crash /tmp/eb-FAO8AO/easybuild-ea15Cq.log
    == processing EasyBuild easyconfig /opt/apps/resif/devel/v1.1-20150414/.installRef/easybuild-easyconfigs/easybuild/easyconfigs/h/HPL/HPL-2.0-goolf-1.4.10.eb
    == building and installing tools/HPL/2.0-goolf-1.4.10...
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
    == packaging...
    == postprocessing...
    == sanity checking...
    == cleaning up...
    == creating module...
    == COMPLETED: Installation ended successfully
    == Results of the build can be found in the log file /home/users/mschmitt/.local/easybuild/software/tools/HPL/2.0-goolf-1.4.10/easybuild/easybuild-HPL-2.0-20150624.113223.log
    == Build succeeded for 1 out of 1
    == temporary log file(s) /tmp/eb-FAO8AO/easybuild-ea15Cq.log* have been removed.
    == temporary directory /tmp/eb-FAO8AO has been removed.

    real    1m10.619s
    user    0m49.387s
    sys     0m7.828s


Let's have a look at `HPL-2.0-ictce-5.3.0` which is not installed yet.
We can check if a software and its dependencies are installed using the `-Dr` option flag:

    $> eb HPL-2.0-ictce-5.3.0.eb -Dr

    == temporary log file in case of crash /tmp/eb-HlZDMR/easybuild-JbndYN.log
    Dry run: printing build status of easyconfigs and dependencies
    CFGS=/opt/apps/resif/devel/v1.1-20150414/.installRef/easybuild-easyconfigs/easybuild/easyconfigs
     * [x] $CFGS/i/icc/icc-2013.3.163.eb (module: compiler/icc/2013.3.163)
     * [x] $CFGS/i/ifort/ifort-2013.3.163.eb (module: compiler/ifort/2013.3.163)
     * [x] $CFGS/i/iccifort/iccifort-2013.3.163.eb (module: toolchain/iccifort/2013.3.163)
     * [x] $CFGS/i/impi/impi-4.1.0.030-iccifort-2013.3.163.eb (module: mpi/impi/4.1.0.030-iccifort-2013.3.163)
     * [x] $CFGS/i/iimpi/iimpi-5.3.0.eb (module: toolchain/iimpi/5.3.0)
     * [x] $CFGS/i/imkl/imkl-11.0.3.163-iimpi-5.3.0.eb (module: numlib/imkl/11.0.3.163-iimpi-5.3.0)
     * [x] $CFGS/i/ictce/ictce-5.3.0.eb (module: toolchain/ictce/5.3.0)
     * [ ] $CFGS/h/HPL/HPL-2.0-ictce-5.3.0.eb (module: tools/HPL/2.0-ictce-5.3.0)
    == temporary log file(s) /tmp/eb-HlZDMR/easybuild-JbndYN.log* have been removed.
    == temporary directory /tmp/eb-HlZDMR has been removed.


`HPL-2.0-ictce-5.3.0` is not available but all it dependencies are. Let's build it:

    $> time eb HPL-2.0-ictce-5.3.0.eb

    == temporary log file in case of crash /tmp/eb-UFlEv7/easybuild-uVbm24.log
    == processing EasyBuild easyconfig /opt/apps/resif/devel/v1.1-20150414/.installRef/easybuild-easyconfigs/easybuild/easyconfigs/h/HPL/HPL-2.0-ictce-5.3.0.eb
    == building and installing tools/HPL/2.0-ictce-5.3.0...
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
    == packaging...
    == postprocessing...
    == sanity checking...
    == cleaning up...
    == creating module...
    == COMPLETED: Installation ended successfully
    == Results of the build can be found in the log file /home/users/mschmitt/.local/easybuild/software/tools/HPL/2.0-ictce-5.3.0/easybuild/easybuild-HPL-2.0-20150624.113547.log
    == Build succeeded for 1 out of 1
    == temporary log file(s) /tmp/eb-UFlEv7/easybuild-uVbm24.log* have been removed.
    == temporary directory /tmp/eb-UFlEv7 has been removed.

    real    1m25.849s
    user    0m49.039s
    sys     0m10.961s


To see the newly installed modules, you need to add the path where they were installed to the MODULEPATH. On the cluster you have to use the `module use` command:

    $> module use $HOME/.local/easybuild/modules/all/

Check which HPL modules are available now:

    $> module avail HPL

    ------------- /mnt/nfs/users/homedirs/mschmitt/.local/easybuild/modules/all -------------
        tools/HPL/2.0-goolf-1.4.10    tools/HPL/2.0-ictce-5.3.0 (D)

    ---------------- /opt/apps/resif/devel/v1.1-20150414/core/modules/tools ----------------
        tools/HPL/2.0-goolf-1.4.10

The two newly-built versions of HPL are now available for your user. You can use them with the usually `module load` command.

### Iris

Let's search for available EasyConfig files with HPL in their name. The EasyConfig files are named with the `.eb` extension.

    $> eb -S HPL

		CFGS1=/home/users/sdiehl/.local/easybuild/software/tools/EasyBuild/3.2.1/lib/python2.7/site-packages/easybuild_easyconfigs-3.2.1-py2.7.egg/easybuild/easyconfigs
		 * $CFGS1/h/HPL/HPL-2.0-foss-2014b.eb
		 * $CFGS1/h/HPL/HPL-2.0-goolf-1.4.10.eb
		 * $CFGS1/h/HPL/HPL-2.0-goolf-1.5.16.eb
		 * $CFGS1/h/HPL/HPL-2.0-ictce-5.3.0.eb
		 * $CFGS1/h/HPL/HPL-2.0-ictce-6.1.5.eb
		 * $CFGS1/h/HPL/HPL-2.1-CrayCCE-2015.06.eb
		 * $CFGS1/h/HPL/HPL-2.1-CrayCCE-2015.11.eb
		 * $CFGS1/h/HPL/HPL-2.1-CrayGNU-2015.06.eb
		 * $CFGS1/h/HPL/HPL-2.1-CrayGNU-2015.11.eb
		 * $CFGS1/h/HPL/HPL-2.1-CrayGNU-2016.03.eb
		 * $CFGS1/h/HPL/HPL-2.1-CrayGNU-2016.04.eb
		 * $CFGS1/h/HPL/HPL-2.1-CrayGNU-2016.06.eb
		 * $CFGS1/h/HPL/HPL-2.1-CrayIntel-2015.06.eb
		 * $CFGS1/h/HPL/HPL-2.1-CrayIntel-2015.11.eb
		 * $CFGS1/h/HPL/HPL-2.1-CrayIntel-2016.06.eb
		 * $CFGS1/h/HPL/HPL-2.1-foss-2015.05.eb
		 * $CFGS1/h/HPL/HPL-2.1-foss-2015a.eb
		 * $CFGS1/h/HPL/HPL-2.1-foss-2015b.eb
		 * $CFGS1/h/HPL/HPL-2.1-foss-2016.04.eb
		 * $CFGS1/h/HPL/HPL-2.1-foss-2016.06.eb
		 * $CFGS1/h/HPL/HPL-2.1-foss-2016a.eb
		 * $CFGS1/h/HPL/HPL-2.1-foss-2016b.eb
		 * $CFGS1/h/HPL/HPL-2.1-gimkl-2.11.5.eb
		 * $CFGS1/h/HPL/HPL-2.1-gmpolf-2016a.eb
		 * $CFGS1/h/HPL/HPL-2.1-gmvolf-1.7.20.eb
		 * $CFGS1/h/HPL/HPL-2.1-gmvolf-2016a.eb
		 * $CFGS1/h/HPL/HPL-2.1-goolf-1.7.20.eb
		 * $CFGS1/h/HPL/HPL-2.1-ictce-7.1.2.eb
		 * $CFGS1/h/HPL/HPL-2.1-ictce-7.3.5.eb
		 * $CFGS1/h/HPL/HPL-2.1-intel-2014.06.eb
		 * $CFGS1/h/HPL/HPL-2.1-intel-2014.10.eb
		 * $CFGS1/h/HPL/HPL-2.1-intel-2014.11.eb
		 * $CFGS1/h/HPL/HPL-2.1-intel-2014b.eb
		 * $CFGS1/h/HPL/HPL-2.1-intel-2015.02.eb
		 * $CFGS1/h/HPL/HPL-2.1-intel-2015.08.eb
		 * $CFGS1/h/HPL/HPL-2.1-intel-2015a.eb
		 * $CFGS1/h/HPL/HPL-2.1-intel-2015b.eb
		 * $CFGS1/h/HPL/HPL-2.1-intel-2016.00.eb
		 * $CFGS1/h/HPL/HPL-2.1-intel-2016.01.eb
		 * $CFGS1/h/HPL/HPL-2.1-intel-2016.02-GCC-4.9.eb
		 * $CFGS1/h/HPL/HPL-2.1-intel-2016.02-GCC-5.3.eb
		 * $CFGS1/h/HPL/HPL-2.1-intel-2016.03-GCC-4.9.eb
		 * $CFGS1/h/HPL/HPL-2.1-intel-2016.03-GCC-5.3.eb
		 * $CFGS1/h/HPL/HPL-2.1-intel-2016.03-GCC-5.4.eb
		 * $CFGS1/h/HPL/HPL-2.1-intel-2016a.eb
		 * $CFGS1/h/HPL/HPL-2.1-intel-2016b.eb
		 * $CFGS1/h/HPL/HPL-2.1-iomkl-2015.01.eb
		 * $CFGS1/h/HPL/HPL-2.1-iomkl-2015.02.eb
		 * $CFGS1/h/HPL/HPL-2.1-iomkl-2015.03.eb
		 * $CFGS1/h/HPL/HPL-2.1-iomkl-2016.07.eb
		 * $CFGS1/h/HPL/HPL-2.1-pomkl-2016.03.eb
		 * $CFGS1/h/HPL/HPL-2.1-pomkl-2016.04.eb
		 * $CFGS1/h/HPL/HPL-2.1-pomkl-2016.09.eb
		 * $CFGS1/h/HPL/HPL-2.1_LINKER-ld.patch
		 * $CFGS1/h/HPL/HPL-2.2-foss-2016.07.eb
		 * $CFGS1/h/HPL/HPL-2.2-foss-2016.09.eb
		 * $CFGS1/h/HPL/HPL-2.2-foss-2017a.eb
		 * $CFGS1/h/HPL/HPL-2.2-goolfc-2016.08.eb
		 * $CFGS1/h/HPL/HPL-2.2-goolfc-2016.10.eb
		 * $CFGS1/h/HPL/HPL-2.2-intel-2017.00.eb
		 * $CFGS1/h/HPL/HPL-2.2-intel-2017.01.eb
		 * $CFGS1/h/HPL/HPL-2.2-intel-2017.02.eb
		 * $CFGS1/h/HPL/HPL-2.2-intel-2017a.eb
		 * $CFGS1/h/HPL/HPL-2.2-intelcuda-2016.10.eb
		 * $CFGS1/h/HPL/HPL-2.2-iomkl-2016.09-GCC-4.9.3-2.25.eb
		 * $CFGS1/h/HPL/HPL-2.2-iomkl-2016.09-GCC-5.4.0-2.26.eb
		 * $CFGS1/h/HPL/HPL-2.2-iomkl-2017.01.eb
		 * $CFGS1/h/HPL/HPL-2.2-iomkl-2017a.eb
		 * $CFGS1/h/HPL/HPL-2.2-pomkl-2016.09.eb
		 * $CFGS1/h/HPL/HPL_parallel-make.patch

		Note: 15 matching archived easyconfig(s) found, use --consider-archived-easyconfigs to see them


Let's have a look at `HPL-2.2-intel-2017a` which is not installed yet.
We can check if a software and its dependencies are installed using the `-Dr` option flag:

    $> eb HPL-2.2-intel-2017a.eb -Dr

		== temporary log file in case of crash /tmp/eb-K1VnEh/easybuild-4C6ZpN.log
		Dry run: printing build status of easyconfigs and dependencies
		CFGS=/home/users/sdiehl/.local/easybuild/software/tools/EasyBuild/3.2.1/lib/python2.7/site-packages/easybuild_easyconfigs-3.2.1-py2.7.egg/easybuild/easyconfigs
		 * [x] $CFGS/m/M4/M4-1.4.17.eb (module: devel/M4/1.4.17)
		 * [x] $CFGS/b/Bison/Bison-3.0.4.eb (module: lang/Bison/3.0.4)
		 * [x] $CFGS/f/flex/flex-2.6.0.eb (module: lang/flex/2.6.0)
		 * [x] $CFGS/z/zlib/zlib-1.2.8.eb (module: lib/zlib/1.2.8)
		 * [x] $CFGS/b/binutils/binutils-2.27.eb (module: tools/binutils/2.27)
		 * [x] $CFGS/g/GCCcore/GCCcore-6.3.0.eb (module: compiler/GCCcore/6.3.0)
		 * [x] $CFGS/m/M4/M4-1.4.18-GCCcore-6.3.0.eb (module: devel/M4/1.4.18-GCCcore-6.3.0)
		 * [x] $CFGS/z/zlib/zlib-1.2.11-GCCcore-6.3.0.eb (module: lib/zlib/1.2.11-GCCcore-6.3.0)
		 * [x] $CFGS/h/help2man/help2man-1.47.4-GCCcore-6.3.0.eb (module: tools/help2man/1.47.4-GCCcore-6.3.0)
		 * [x] $CFGS/b/Bison/Bison-3.0.4-GCCcore-6.3.0.eb (module: lang/Bison/3.0.4-GCCcore-6.3.0)
		 * [x] $CFGS/f/flex/flex-2.6.3-GCCcore-6.3.0.eb (module: lang/flex/2.6.3-GCCcore-6.3.0)
		 * [x] $CFGS/b/binutils/binutils-2.27-GCCcore-6.3.0.eb (module: tools/binutils/2.27-GCCcore-6.3.0)
		 * [x] $CFGS/i/icc/icc-2017.1.132-GCC-6.3.0-2.27.eb (module: compiler/icc/2017.1.132-GCC-6.3.0-2.27)
		 * [x] $CFGS/i/ifort/ifort-2017.1.132-GCC-6.3.0-2.27.eb (module: compiler/ifort/2017.1.132-GCC-6.3.0-2.27)
		 * [x] $CFGS/i/iccifort/iccifort-2017.1.132-GCC-6.3.0-2.27.eb (module: toolchain/iccifort/2017.1.132-GCC-6.3.0-2.27)
		 * [x] $CFGS/i/impi/impi-2017.1.132-iccifort-2017.1.132-GCC-6.3.0-2.27.eb (module: mpi/impi/2017.1.132-iccifort-2017.1.132-GCC-6.3.0-2.27)
		 * [x] $CFGS/i/iimpi/iimpi-2017a.eb (module: toolchain/iimpi/2017a)
		 * [x] $CFGS/i/imkl/imkl-2017.1.132-iimpi-2017a.eb (module: numlib/imkl/2017.1.132-iimpi-2017a)
		 * [x] $CFGS/i/intel/intel-2017a.eb (module: toolchain/intel/2017a)
		 * [ ] $CFGS/h/HPL/HPL-2.2-intel-2017a.eb (module: tools/HPL/2.2-intel-2017a)
		== Temporary log file(s) /tmp/eb-K1VnEh/easybuild-4C6ZpN.log* have been removed.
		== Temporary directory /tmp/eb-K1VnEh has been removed.


`HPL-2.2-intel-2017a` is not available but all it dependencies are. Let's build it:

    $> time eb HPL-2.2-intel-2017a.eb

		== temporary log file in case of crash /tmp/eb-152mYB/easybuild-myA4bD.log
		== processing EasyBuild easyconfig /home/users/sdiehl/.local/easybuild/software/tools/EasyBuild/3.2.1/lib/python2.7/site-packages/easybuild_easyconfigs-3.2.1-py2.7.egg/easybuild/easyconfigs/h/HPL/HPL-2.2-intel-2017a.eb
		== building and installing tools/HPL/2.2-intel-2017a...
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
		== postprocessing...
		== sanity checking...
		== cleaning up...
		== creating module...
		== permissions...
		== packaging...
		== COMPLETED: Installation ended successfully
		== Results of the build can be found in the log file(s) /home/users/sdiehl/.local/easybuild/software/tools/HPL/2.2-intel-2017a/easybuild/easybuild-HPL-2.2-20170609.155430.log
		== Build succeeded for 1 out of 1
		== Temporary log file(s) /tmp/eb-152mYB/easybuild-myA4bD.log* have been removed.
		== Temporary directory /tmp/eb-152mYB has been removed.

		real	0m54.624s
		user	0m14.651s
		sys	0m21.476s


To see the newly installed modules, you need to add the path where they were installed to the MODULEPATH. On the cluster you have to use the `module use` command:

    $> module use $HOME/.local/easybuild/modules/all

Check which HPL modules are available now:

    $> module avail HPL

		---------------------- /home/users/sdiehl/.local/easybuild/modules/all ----------------------
		   tools/HPL/2.2-intel-2017a


The newly-built version of HPL is now available for your user. You can use them with the usually `module load` command.

## Amending an existing EasyConfig file (Gaia only)

It is possible to amend existing EasyConfig file to build software with slightly different parameters.

As a example, we are going to build the lastest version of HPL (2.1) with ICTCE toolchain. We use the `--try-software-version` option flag to overide the HPL version.

    $> time eb HPL-2.0-ictce-5.3.0.eb --try-software-version=2.1

    == temporary log file in case of crash /tmp/eb-ocChbK/easybuild-liMmlk.log
    == processing EasyBuild easyconfig /tmp/eb-ocChbK/tweaked_easyconfigs/HPL-2.1-ictce-5.3.0.eb
    == building and installing tools/HPL/2.1-ictce-5.3.0...
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
    == packaging...
    == postprocessing...
    == sanity checking...
    == cleaning up...
    == creating module...
    == COMPLETED: Installation ended successfully
    == Results of the build can be found in the log file /home/users/mschmitt/.local/easybuild/software/tools/HPL/2.1-ictce-5.3.0/easybuild/easybuild-HPL-2.1-20150624.114243.log
    == Build succeeded for 1 out of 1
    == temporary log file(s) /tmp/eb-ocChbK/easybuild-liMmlk.log* have been removed.
    == temporary directory /tmp/eb-ocChbK has been removed.

    real    1m24.933s
    user    0m53.167s
    sys     0m11.533s

    $> module avail HPL

    ------------- /mnt/nfs/users/homedirs/mschmitt/.local/easybuild/modules/all -------------
        tools/HPL/2.0-goolf-1.4.10    tools/HPL/2.0-ictce-5.3.0    tools/HPL/2.1-ictce-5.3.0 (D)

    ---------------- /opt/apps/resif/devel/v1.1-20150414/core/modules/tools ----------------
        tools/HPL/2.0-goolf-1.4.10

We obtained HPL 2.1 without writing any EasyConfig file.

**IMPORTANT**: LMod cache the modules available such that it may append that the `module avail HPL` command _does not_ report the newly created 2.1 version. In that case, you can use the following option:

    $> module --ignore-cache avail HPL

There are multiple ways to amend a EasyConfig file. Check the `--try-*` option flags for all the possibilities.


## Build software using your own EasyConfig file (Gaia only)

For this example, we create an EasyConfig file to build GZip 1.4 with the GOOLF toolchain.
Open your favorite editor and create a file named `gzip-1.4-goolf-1.4.10.eb` with the following content:

    easyblock = 'ConfigureMake'

    name = 'gzip'
    version = '1.4'

    homepage = 'http://www.gnu.org/software/gzip/'
    description = "gzip (GNU zip) is a popular data compression program as a replacement for compress"

    # use the GOOLF toolchain
    toolchain = {'name': 'goolf', 'version': '1.4.10'}

    # specify that GCC compiler should be used to build gzip
    preconfigopts = "CC='gcc'"

    # source tarball filename
    sources = ['%s-%s.tar.gz'%(name,version)]

    # download location for source files
    source_urls = ['http://ftpmirror.gnu.org/gzip']

    # make sure the gzip and gunzip binaries are available after installation
    sanity_check_paths = {
                          'files': ["bin/gunzip", "bin/gzip"],
                          'dirs': []
                         }

    # run 'gzip -h' and 'gzip --version' after installation
    sanity_check_commands = [True, ('gzip', '--version')]


This is a simple EasyConfig. Most of the fields are self-descriptive. No build method is explicitely defined, so it uses by default the standard *configure/make/make install* approach.


Let's build GZip with this EasyConfig file:

    $> time eb gzip-1.4-goolf-1.4.10.eb

    == temporary log file in case of crash /tmp/eb-hiyyN1/easybuild-ynLsHC.log
    == processing EasyBuild easyconfig /mnt/nfs/users/homedirs/mschmitt/gzip-1.4-goolf-1.4.10.eb
    == building and installing base/gzip/1.4-goolf-1.4.10...
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
    == packaging...
    == postprocessing...
    == sanity checking...
    == cleaning up...
    == creating module...
    == COMPLETED: Installation ended successfully
    == Results of the build can be found in the log file /home/users/mschmitt/.local/easybuild/software/base/gzip/1.4-goolf-1.4.10/easybuild/easybuild-gzip-1.4-20150624.114745.log
    == Build succeeded for 1 out of 1
    == temporary log file(s) /tmp/eb-hiyyN1/easybuild-ynLsHC.log* have been removed.
    == temporary directory /tmp/eb-hiyyN1 has been removed.

    real    1m39.982s
    user    0m52.743s
    sys     0m11.297s


We can now check that our version of GZip is available via the modules:

    $> module avail gzip

    --------- /mnt/nfs/users/homedirs/mschmitt/.local/easybuild/modules/all ---------
        base/gzip/1.4-goolf-1.4.10



## To go further


- [EasyBuild homepage](http://hpcugent.github.io/easybuild)
- [EasyBuild documentation](http://hpcugent.github.io/easybuild/)
- [Getting started](https://github.com/hpcugent/easybuild/wiki/Getting-started)
- [Using EasyBuild](https://github.com/hpcugent/easybuild/wiki/Using-EasyBuild)
- [Step-by-step guide](https://github.com/hpcugent/easybuild/wiki/Step-by-step-guide)
