`README.md`

Copyright (c) 2014 Xavier Besseron <xavier.besseron@uni.lu>


-------------------

# UL HPC Tutorial: Build software with EasyBuild on UL HPC platform

The objective of this tutorial is to TODO  
on UL HPC platform.

Ensure you are able to [connect to the chaos and gaia cluster](https://hpc.uni.lu/users/docs/access.html). 

	/!\ FOR ALL YOUR COMPILATION WITH EASYBUILD, ENSURE YOU WORK ON A COMPUTING NODE
	
	(access)$> 	oarsub -I -l nodes=1,walltime=4

The latest version of this tutorial is available on
[Github](https://github.com/ULHPC/tutorials/tree/devel/advanced/EasyBuild)

## Introduction to EasyBuild

TODO
use cases:
* build supported software (>XXXX)
* 

build and install at user level (ie in your home directory), don't need the admin


### EasyBuild concepts


**Toolchains** and **EasyConfig files** 





## EasyBuild on UL HPC platform

To use EasyBuild on a compute node, load the EasyBuild module:


    $> module avail EasyBuild
        
    ----------------------- /opt/apps/HPCBIOS/modules/tools ------------------------
    EasyBuild/1.10.0

    $> module load EasyBuild/1.10.0  # or any other version you prefer
    

The EasyBuild command is `eb`. Check the version you have loaded:

    $> eb --version
    
    This is EasyBuild 1.10.0 (framework: 1.10.0, easyblocks: 1.10.0) on host gaia-59.
    




## Build using provided EasyConfig file

In this part, we propose to be High Performance Linpack (HPL) using EasyBuild. 
HPL is supported by EasyBuild, this means that an EasyConfig file allowing to build HPL is already provided with EasyBuild.


First, let's see which HPL are available on the cluster:

    $> module avail HPL
    
    ----------------------- /opt/apps/HPCBIOS/modules/tools ------------------------
    HPL/2.0-cgmpolf-1.1.6       HPL/2.0-goolf-1.4.10
    HPL/2.0-cgmvolf-1.2.7       HPL/2.0-ictce-4.0.6
    HPL/2.0-cgoolf-1.1.7        HPL/2.0-ictce-5.3.0
    HPL/2.0-goalf-1.1.0-no-OFED
    

Then, search for available EasyConfig files with HPL in their name:

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


However the build can be forced using the `-f` option flag:

    $> eb HPL-2.0-goolf-1.4.10.eb
    
    TODO

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
    HPL/2.0-goolf-1.4.10        HPL/2.0-ictce-6.1.5

    ----------------------- /opt/apps/HPCBIOS/modules/tools ------------------------
    HPL/2.0-cgmpolf-1.1.6       HPL/2.0-goolf-1.4.10
    HPL/2.0-cgmvolf-1.2.7       HPL/2.0-ictce-4.0.6
    HPL/2.0-cgoolf-1.1.7        HPL/2.0-ictce-5.3.0
    HPL/2.0-goalf-1.1.0-no-OFED
    

The two newly-built versions of HPL are now available for your user. You can use them with the usually `module load` command.


    

## Amending an existing EasyConfig file



## Build a software using your own EasyConfig file


## To go further

