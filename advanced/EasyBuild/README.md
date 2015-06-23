`README.md`

Copyright (c) 2014 Xavier Besseron <xavier.besseron@uni.lu>


-------------------

# UL HPC Tutorial: Build software with EasyBuild on UL HPC platform

The objective of this tutorial is to show how EasyBuild can be used to ease, automate and script the build of software on the UL HPC platforms. 

Two use-cases are considered. First, we are going to build software that are supported by EasyBuild. In a second time, we will see through a simple example how to add support for a new software in EasyBuild.

The benefit of using EasyBuild for your builds is that it allows automated and reproducable build of software. Once a build has been made, the build script (via the *EasyConfig file*) or the installed software (via the *module file*) can be shared with other users.

Before starting this tutorial, ensure you are able to [connect to the chaos and gaia cluster](https://hpc.uni.lu/users/docs/access.html). 
**For all your compilation with Easybuild, you must work on a computing node:**
	
	(access)$> 	oarsub -I -l nodes=1,walltime=4


The latest version of this tutorial is available on our
[readthedocs](http://ulhpc-tutorials.readthedocs.org/en/latest/advanced/easybuild/) documentation.

## Short introduction to EasyBuild

EasyBuild is a tool that allows to perform automated and reproducible compilation and installation of software. A large number of scientific software are supported (479 software packages in the last release).

All builds and installations are performed at user level, so you don't need the admin rights. 
The software are installed in your home directory (by default in `$HOME/.local/easybuild/software/`) and a module file is generated (by default in `$HOME/.local/easybuild/modules/`) to use the software.

EasyBuild relies on two main concepts: *Toolchains* and *EasyConfig file*.

A **toolchain** corresponds to a compiler and a set of libraries which are commonly used to build a software. The two main toolchains frequently used on the UL HPC platform are the GOOLF and the ICTCE toolchains. GOOLF is based on the GCC compiler and on open-source libraries (OpenMPI, OpenBLAS, etc.). ICTCE is based on the Intel compiler and on Intel libraries (Intel MPI, Intel Math Kernel Library, etc.). 

An **EasyConfig file** is a simple text file that describes the build process of a software. For most software that uses standard procedure (like `configure`, `make` and `make install`), this file is very simple. Many EasyConfig files are already provided with EasyBuild.


EasyConfig files and generated modules are named using the following convention:
`<Software-Name>-<Software-Version>-<Toolchain-Name>-<Toolchain-Version>` 

Additional details are available on EasyBuild website:

- [EasyBuild homepage](http://hpcugent.github.io/easybuild)
- [EasyBuild documentation](http://hpcugent.github.io/easybuild/)
- [What is EasyBuild?](https://github.com/hpcugent/easybuild/wiki/EasyBuild)
- [Toolchains](https://github.com/hpcugent/easybuild/wiki/Compiler-toolchains)
- [EasyConfig files](https://github.com/hpcugent/easybuild/wiki/Easyconfig-files)
- [List of supported software packages](https://github.com/hpcugent/easybuild/wiki/List-of-supported-software-packages)


## EasyBuild on UL HPC platform

To use EasyBuild on a compute node, load the EasyBuild module:


    $> module avail EasyBuild
        
    ----------------------- /opt/apps/HPCBIOS/modules/tools ------------------------
    EasyBuild/1.10.0

    $> module load EasyBuild/1.10.0  # or any other version you prefer
    

The EasyBuild command is `eb`. Check the version you have loaded:

    $> eb --version
    
    This is EasyBuild 1.10.0 (framework: 1.10.0, easyblocks: 1.10.0) on host gaia-59.
    

To get help on the EasyBuild options, use the `-h` or `-H` option flags:

    $> eb -h
    $> eb -H
    



## Build software using provided EasyConfig file

In this part, we propose to be High Performance Linpack (HPL) using EasyBuild. 
HPL is supported by EasyBuild, this means that an EasyConfig file allowing to build HPL is already provided with EasyBuild.


First, let's see which HPL are available on the cluster:

    $> module avail HPL
    
    ----------------------- /opt/apps/HPCBIOS/modules/tools ------------------------
    HPL/2.0-cgmpolf-1.1.6       HPL/2.0-goolf-1.4.10
    HPL/2.0-cgmvolf-1.2.7       HPL/2.0-ictce-4.0.6
    HPL/2.0-cgoolf-1.1.7        HPL/2.0-ictce-5.3.0
    HPL/2.0-goalf-1.1.0-no-OFED
    

Then, search for available EasyConfig files with HPL in their name. The EasyConfig files are named with the `.eb` extension.

    $> eb -S HPL
    
    == temporary log file in case of crash /tmp/easybuild-TJBo27/easybuild-0xj0cG.log
    == Searching (case-insensitive) for 'HPL' in /opt/apps/HPCBIOS.20131224/software/EasyBuild/1.10.0/lib/python2.6/site-packages/easybuild_easyconfigs-1.10.0.0-py2.6.egg/easybuild/easyconfigs 
    CFGS1=/opt/apps/HPCBIOS.20131224/software/EasyBuild/1.10.0/lib/python2.6/site-packages/easybuild_easyconfigs-1.10.0.0-py2.6.egg/easybuild/easyconfigs/h/HPL
     * $CFGS1/HPL-2.0-cgmpolf-1.1.6.eb
     * $CFGS1/HPL-2.0-cgmvolf-1.1.12rc1.eb
     * $CFGS1/HPL-2.0-cgmvolf-1.2.7.eb
     * $CFGS1/HPL-2.0-cgoolf-1.1.7.eb
     * $CFGS1/HPL-2.0-goalf-1.1.0-no-OFED.eb
     * $CFGS1/HPL-2.0-goolf-1.4.10.eb
     * $CFGS1/HPL-2.0-ictce-4.0.6.eb
     * $CFGS1/HPL-2.0-ictce-5.3.0.eb
     * $CFGS1/HPL-2.0-ictce-6.0.5.eb
     * $CFGS1/HPL-2.0-ictce-6.1.5.eb
     * $CFGS1/HPL-2.0-iomkl-4.6.13.eb
     * $CFGS1/HPL_parallel-make.patch
    == temporary log file /tmp/easybuild-TJBo27/easybuild-0xj0cG.log has been removed.
    == temporary directory /tmp/easybuild-TJBo27 has been removed.
    

If we try to build `HPL-2.0-goolf-1.4.10`, nothing will be done as it is already installed on the cluster.

    $> eb HPL-2.0-goolf-1.4.10.eb

    == temporary log file in case of crash /tmp/easybuild-4Cy8Qn/easybuild-yRhXIT.log
    == HPL/2.0-goolf-1.4.10 is already installed (module found), skipping
    == No easyconfigs left to be built.
    == Build succeeded for 0 out of 0
    == temporary log file /tmp/easybuild-4Cy8Qn/easybuild-yRhXIT.log has been removed.
    == temporary directory /tmp/easybuild-4Cy8Qn has been removed.


However the build can be forced using the `-f` option flag. Then this software will be re-built.
(Tip: prefix your command with `time` to know its duration)

    $> time eb HPL-2.0-goolf-1.4.10.eb -f
    
    == temporary log file in case of crash /tmp/easybuild-FWc7Dl/easybuild-MEhAw4.log
    == resolving dependencies ...
    == processing EasyBuild easyconfig /opt/apps/HPCBIOS.20131224/software/EasyBuild/1.10.0/lib/python2.6/site-packages/easybuild_easyconfigs-1.10.0.0-py2.6.egg/easybuild/easyconfigs/h/HPL/HPL-2.0-goolf-1.4.10.eb
    == building and installing HPL/2.0-goolf-1.4.10...
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
    == 
    WARNING: Build exited with exit code 0. 2 possible error(s) were detected in the build logs, please verify the build.
    
    == Results of the build can be found in the log file /home/users/xbesseron/.local/easybuild/software/HPL/2.0-goolf-1.4.10/easybuild/easybuild-HPL-2.0-20140505.195534.log
    == Build succeeded for 1 out of 1
    == temporary log file /tmp/easybuild-FWc7Dl/easybuild-MEhAw4.log has been removed.
    == temporary directory /tmp/easybuild-FWc7Dl has been removed.
    
    real    0m18.573s
    user    0m9.689s
    sys     0m3.936s
    
    

Let's have a look at `HPL-2.0-ictce-6.1.5` which is not installed yet. 
We can check if a software and its dependencies are installed using the `-Dr` option flag:

    $> eb HPL-2.0-ictce-6.1.5.eb -Dr
    
    == temporary log file in case of crash /tmp/easybuild-SJb9CJ/easybuild-YJMJjy.log
    Dry run: printing build status of easyconfigs and dependencies
    CFGS=/opt/apps/HPCBIOS.20131224/software/EasyBuild/1.10.0/lib/python2.6/site-packages/easybuild_easyconfigs-1.10.0.0-py2.6.egg/easybuild/easyconfigs
     * [x] $CFGS/i/icc/icc-2013_sp1.1.106.eb (module: icc/2013_sp1.1.106)
     * [x] $CFGS/i/ifort/ifort-2013_sp1.1.106.eb (module: ifort/2013_sp1.1.106)
     * [x] $CFGS/i/impi/impi-4.1.3.045.eb (module: impi/4.1.3.045)
     * [x] $CFGS/i/imkl/imkl-11.1.1.106.eb (module: imkl/11.1.1.106)
     * [x] $CFGS/i/ictce/ictce-6.1.5.eb (module: ictce/6.1.5)
     * [ ] $CFGS/h/HPL/HPL-2.0-ictce-6.1.5.eb (module: HPL/2.0-ictce-6.1.5)
    == temporary log file /tmp/easybuild-SJb9CJ/easybuild-YJMJjy.log has been removed.
    == temporary directory /tmp/easybuild-SJb9CJ has been removed.
    

`HPL-2.0-ictce-6.1.5` is not available but all it dependencies are. Let's build it:


    $> time eb HPL-2.0-ictce-6.1.5.eb
    
    == temporary log file in case of crash /tmp/easybuild-Vo8xZe/easybuild-IT7Abp.log
    == resolving dependencies ...
    == processing EasyBuild easyconfig /opt/apps/HPCBIOS.20131224/software/EasyBuild/1.10.0/lib/python2.6/site-packages/easybuild_easyconfigs-1.10.0.0-py2.6.egg/easybuild/easyconfigs/h/HPL/HPL-2.0-ictce-6.1.5.eb
    == building and installing HPL/2.0-ictce-6.1.5...
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
    == 
    WARNING: Build exited with exit code 0. 2 possible error(s) were detected in the build logs, please verify the build.
    
    == Results of the build can be found in the log file /home/users/xbesseron/.local/  easybuild/software/HPL/2.0-ictce-6.1.5/easybuild/easybuild-HPL-2.0-20140430.151555.log
    == Build succeeded for 1 out of 1
    == temporary log file /tmp/easybuild-Vo8xZe/easybuild-IT7Abp.log has been removed.
    == temporary directory /tmp/easybuild-Vo8xZe has been removed.
    
    real    1m3.592s
    user    0m12.153s
    sys     0m6.140s



Check which HPL modules are available now:

    $> module avail HPL
    
    -------------- /home/users/xbesseron/.local/easybuild/modules/all --------------
    HPL/2.0-goolf-1.4.10 HPL/2.0-ictce-6.1.5

    ----------------------- /opt/apps/HPCBIOS/modules/tools ------------------------
    HPL/2.0-cgmpolf-1.1.6       HPL/2.0-goolf-1.4.10
    HPL/2.0-cgmvolf-1.2.7       HPL/2.0-ictce-4.0.6
    HPL/2.0-cgoolf-1.1.7        HPL/2.0-ictce-5.3.0
    HPL/2.0-goalf-1.1.0-no-OFED
    

The two newly-built versions of HPL are now available for your user. You can use them with the usually `module load` command.


    

## Amending an existing EasyConfig file

It is possible to amend existing EasyConfig file to build software with slightly different parameters. 

As a example, we are going to build the lastest version of HPL (2.1) with ICTCE toolchain. We use the `--try-software-version` option flag to overide the HPL version.

    $> time eb HPL-2.0-ictce-6.1.5.eb --try-software-version=2.1
    
    == temporary log file in case of crash /tmp/easybuild-182xZg/easybuild-5dnc25.log
    == resolving dependencies ...
    == processing EasyBuild easyconfig /tmp/easybuild-182xZg/HPL-2.1-ictce-6.1.5.eb
    == building and installing HPL/2.1-ictce-6.1.5...
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
    == Results of the build can be found in the log file /home/users/xbesseron/.local/easybuild/   software/HPL/2.1-ictce-6.1.5/easybuild/easybuild-HPL-2.1-20140430.151656.log
    == Build succeeded for 1 out of 1
    == temporary log file /tmp/easybuild-182xZg/easybuild-5dnc25.log has been removed.
    == temporary directory /tmp/easybuild-182xZg has been removed.

    real    0m59.229s
    user    0m12.305s
    sys     0m5.824s

    $> module avail HPL
    
    -------------- /home/users/xbesseron/.local/easybuild/modules/all --------------
    HPL/2.0-goolf-1.4.10 HPL/2.0-ictce-6.1.5
    HPL/2.1-ictce-6.1.5
    
    ----------------------- /opt/apps/HPCBIOS/modules/tools ------------------------
    HPL/2.0-cgmpolf-1.1.6       HPL/2.0-goolf-1.4.10
    HPL/2.0-cgmvolf-1.2.7       HPL/2.0-ictce-4.0.6
    HPL/2.0-cgoolf-1.1.7        HPL/2.0-ictce-5.3.0
    HPL/2.0-goalf-1.1.0-no-OFED


We obtained HPL 2.1 without writing any EasyConfig file.

There are multiple ways to amend a EasyConfig file. Check the `--try-*` option flags for all the possibilities.


## Build software using your own EasyConfig file


For this example, we create an EasyConfig file to build GZip 1.4 with the GOOLF toolchain.
Open your favorite editor and create a file named `gzip-1.4-goolf-1.4.10.eb` with the following content:

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

    == temporary log file in case of crash /tmp/easybuild-XYUFBw/easybuild-QXd_vb.log
    == resolving dependencies ...
    == processing EasyBuild easyconfig /mnt/nfs/users/homedirs/xbesseron/gzip-1.4-goolf-1.4.10.eb
    == building and installing gzip/1.4-goolf-1.4.10...
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
    == Results of the build can be found in the log file /home/users/xbesseron/.local/easybuild/software/gzip/1.4-goolf-1.4.10/easybuild/easybuild-gzip-1.4-20140430.151818.log
    == Build succeeded for 1 out of 1
    == temporary log file /tmp/easybuild-XYUFBw/easybuild-QXd_vb.log has been removed.
    == temporary directory /tmp/easybuild-XYUFBw has been removed.
    
    real    1m20.706s
    user    0m12.293s
    sys     0m5.912s



We can now check that our version of GZip is available via the modules:

    $> module avail gzip

    -------------- /home/users/xbesseron/.local/easybuild/modules/all --------------
    gzip/1.4-goolf-1.4.10

    ----------------------- /opt/apps/HPCBIOS/modules/tools ------------------------
    gzip/1.5-cgoolf-1.1.7 gzip/1.5-ictce-5.3.0  gzip/1.6-ictce-5.3.0



## To go further


- [EasyBuild homepage](http://hpcugent.github.io/easybuild)
- [EasyBuild documentation](http://hpcugent.github.io/easybuild/)
- [Getting started](https://github.com/hpcugent/easybuild/wiki/Getting-started)
- [Using EasyBuild](https://github.com/hpcugent/easybuild/wiki/Using-EasyBuild)
- [Step-by-step guide](https://github.com/hpcugent/easybuild/wiki/Step-by-step-guide)
