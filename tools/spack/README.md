[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/)[![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/tools/easybuild/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/tools/easybuild/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# The Spack package manager on the UL HPC platform

Copyright (c) 2014-2023 UL HPC Team  <hpc-sysadmins@uni.lu>

Authors: Emmanuel Kieffer and UL HPC Team <hpc-team@uni.lu>

[![Spack](https://spack.io/assets/images/spack-logo.svg)](https://spack.io/)

Like [EasyBuild](http://easybuild.readthedocs.io), [Spack](https://github.com/spack/spack) can be used to install softwares on the [UL HPC](https://hpc.uni.lu) platforms. Spack is a multi platform package manager that builds and installs multiple versions and configurations of software. Spack resolve dependencies and install them like any other package manager you can find on linux platform.

The definition provided by the official documentation is the following one:
<br/>

<i>"Spack is a multi-platform package manager that builds and installs multiple versions and configurations of software. It works on Linux, macOS, and many supercomputers. Spack is non-destructive: installing a new version of a package does not break existing installations, so many configurations of the same package can coexist.

Spack offers a simple "spec" syntax that allows users to specify versions and configuration options. Package files are written in pure Python, and specs allow package authors to write a single script for many different builds of the same package. With Spack, you can build your software all the ways you want to".</i>

--------------------
## Pre-requisites ##

Ensure you are able to [connect to the UL HPC clusters](https://hpc-docs.uni.lu/connect/access/).



Now you'll need to pull the latest changes in your working copy of the [ULHPC/tutorials](https://github.com/ULHPC/tutorials) you should have cloned in `~/git/github.com/ULHPC/tutorials` (see ["preliminaries" tutorial](../../preliminaries/))

``` bash
(access)$ export SPACK_ROOT=${HOME}/.spack
(access)$ git clone -c feature.manyFiles=true https://github.com/spack/spack.git ${SPACK_ROOT}
(access)$ cat  << EOF >> ${HOME}/.bashrc
export SPACK_ROOT=\${HOME}/.spack
if [[ -f \${SPACK_ROOT}/share/spack/setup-env.sh && -n \${SLURM_JOB_ID} ]]; then
    source \${SPACK_ROOT}/share/spack/setup-env.sh
fi
EOF
```

Now, it's time to get a interative job allocation

```bash
(access)$> si --ntasks-per-node 1 -c 4 -t 2:00:00
# ... or using the HPC School reservation 'hpcschool'if needed  - use 'sinfo -T' to check if active and its name
# (access)$> si --reservation=hpcschool --ntasks-per-node 1 -c 4 -t 2:00:00
# Double check if Spack is loaded
(node)$> 
spack
usage: spack [-hkV] [--color {always,never,auto}] COMMAND ...

A flexible package manager that supports multiple versions,
configurations, platforms, and compilers.

These are common spack commands:

query packages:
  list                  list and search available packages
  info                  get detailed information on a particular package
  find                  list and search installed packages

build packages:
  install               build and install packages
  uninstall             remove installed packages
  gc                    remove specs that are now no longer needed
  spec                  show what would be installed, given a spec

configuration:
  external              manage external packages in Spack configuration

environments:
  env                   manage virtual environments
  view                  project packages to a compact naming scheme on the filesystem.

create packages:
  create                create a new package file
  edit                  open package files in $EDITOR

system:
  arch                  print architecture information about this machine
  audit                 audit configuration files, packages, etc.
  compilers             list available compilers

user environment:
  load                  add package to the user environment
  module                generate/manage module files
  unload                remove package from the user environment

optional arguments:
  --color {always,never,auto}
                        when to colorize output (default: auto)
  -V, --version         show version number and exit
  -h, --help            show this help message and exit
  -k, --insecure        do not check ssl certificates when downloading

more help:
  spack help --all       list all commands and options
  spack help <command>   help on a specific command
  spack help --spec      help on the package specification syntax
  spack docs             open https://spack.rtfd.io/ in a browser
```
**For all tests and compilation with Spack, you MUST work on a computing node.**



------------------------------------------
## Simple package installation with Spack

Installing the last version of a package is as simple as :`(node)$ spack install zlib`. This will install the most recent version of zlib. You can load it with `(node)$ spack load zlib`

Spack becomes more complex and more complete when you need to install a specific version with specific dependencies. In fact, you have a large number of possible combinations helping you to be more reproducible.

Some of [Spack's features](https://spack.readthedocs.io/en/latest/features.html#) are:

1. Custom versions & configurations
2. Custom dependencies
3. Non-destructive installs
4. Packages can peacefully coexist
5. Creating packages is easy
6. Virtual environments

* In this tutorial, we will only consider package installation with custom versions and dependencies. Before starting, we will configure spack to use `/dev/shm` as build directory to improve compilation performance. 
* In order to do it, please open `${SPACK_ROOT}/etc/spack/defaults/config.yaml` with you favorite text editor, e.g., vim, emacs, nano.
* Go to the `build_cache` section (~ line 69) and :
  - add first `- /dev/shm/$user/spack-stage`
  - comment `- $user_cache_path/stage`

Now, Spack will try to use `/dev/shm` as first build cache directory. You should see something like this:

```yaml
  build_stage:
    - /dev/shm/$user/spack-stage
    - $tempdir/$user/spack-stage
  #  - $user_cache_path/stage
  # - $spack/var/spack/stage
```

Last check, use the command `spack config get config | grep "\/dev\/shm" ` to ensure that the configuration has been updated accordingly. 


### Part 1: Custom versions and configurations

* The spack command `spack help --spec` is a very useful command to help you remember how to configure your package installation.

* First, check if the package is not already installed with `spack find -vpl <package>`. If not, you can check if Spack's available packages with `spack list <package>`

```bash
(node)$ spack find -vpl hdf5
==> No package matches the query: hdf5
(node)$ spack list hdf5  
hdf5  hdf5-blosc  hdf5-vfd-gds  hdf5-vol-async  hdf5-vol-cache  hdf5-vol-external-passthrough  hdf5-vol-log  r-hdf5array  r-hdf5r  r-rhdf5  r-rhdf5filters  r-rhdf5lib
==> 12 packages
```

Once you find the package listed in the output of the `spack list` command, you can get more info on the package, e.g., hdf5

```bash
(node)$ spack info hdf5
CMakePackage:   hdf5

Description:
    HDF5 is a data model, library, and file format for storing and managing
    data. It supports an unlimited variety of datatypes, and is designed for
    flexible and efficient I/O and for high volume and complex data.

Homepage: https://portal.hdfgroup.org

Preferred version:
    1.14.0           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.14/hdf5-1.14.0/src/hdf5-1.14.0.tar.gz

Safe versions:
    develop-1.15     [git] https://github.com/HDFGroup/hdf5.git on branch develop
    develop-1.14     [git] https://github.com/HDFGroup/hdf5.git on branch hdf5_1_14
    develop-1.12     [git] https://github.com/HDFGroup/hdf5.git on branch hdf5_1_12
    develop-1.10     [git] https://github.com/HDFGroup/hdf5.git on branch hdf5_1_10
    develop-1.8      [git] https://github.com/HDFGroup/hdf5.git on branch hdf5_1_8
    1.14.0           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.14/hdf5-1.14.0/src/hdf5-1.14.0.tar.gz
    1.13.3           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.13/hdf5-1.13.3/src/hdf5-1.13.3.tar.gz
    1.13.2           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.13/hdf5-1.13.2/src/hdf5-1.13.2.tar.gz
    1.12.2           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.2/src/hdf5-1.12.2.tar.gz
    1.12.1           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.1/src/hdf5-1.12.1.tar.gz
    1.12.0           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.0/src/hdf5-1.12.0.tar.gz
    1.10.9           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.9/src/hdf5-1.10.9.tar.gz
    1.10.8           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.8/src/hdf5-1.10.8.tar.gz
    1.10.7           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.7/src/hdf5-1.10.7.tar.gz
    1.10.6           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.6/src/hdf5-1.10.6.tar.gz
    1.10.5           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz
    1.10.4           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.4/src/hdf5-1.10.4.tar.gz
    1.10.3           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.3/src/hdf5-1.10.3.tar.gz
    1.10.2           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.2/src/hdf5-1.10.2.tar.gz
    1.10.1           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.1/src/hdf5-1.10.1.tar.gz
    1.10.0-patch1    https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.0-patch1/src/hdf5-1.10.0-patch1.tar.gz
    1.10.0           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.0/src/hdf5-1.10.0.tar.gz
    1.8.22           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.22/src/hdf5-1.8.22.tar.gz
    1.8.21           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.21/src/hdf5-1.8.21.tar.gz
    1.8.19           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.19/src/hdf5-1.8.19.tar.gz
    1.8.18           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.18/src/hdf5-1.8.18.tar.gz
    1.8.17           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.17/src/hdf5-1.8.17.tar.gz
    1.8.16           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.16/src/hdf5-1.8.16.tar.gz
    1.8.15           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.15/src/hdf5-1.8.15.tar.gz
    1.8.14           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.14/src/hdf5-1.8.14.tar.gz
    1.8.13           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.13/src/hdf5-1.8.13.tar.gz
    1.8.12           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.12/src/hdf5-1.8.12.tar.gz
    1.8.10           https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.10/src/hdf5-1.8.10.tar.gz

Deprecated versions:
    None

Variants:
    Name [Default]                 When                              Allowed values          Description
    ===========================    ==============================    ====================    ============================================

    api [default]                  --                                default, v116, v114,    Choose api compatibility for earlier version
                                                                     v112, v110, v18, v16
    build_system [cmake]           --                                cmake                   Build systems supported by the package
    build_type [RelWithDebInfo]    [build_system=cmake]              Debug, Release,         CMake build type
                                                                     RelWithDebInfo,
                                                                     MinSizeRel
    cxx [off]                      --                                on, off                 Enable C++ support
    fortran [off]                  --                                on, off                 Enable Fortran support
    generator [make]               [build_system=cmake]              ninja,make              the build system generator to use
    hl [off]                       --                                on, off                 Enable the high-level library
    ipo [off]                      [build_system=cmake               on, off                 CMake interprocedural optimization
                                   ^cmake@3.9:]
    java [off]                     [@1.10:]                          on, off                 Enable Java support
    map [off]                      [@1.14:]                          on, off                 Enable MAP API support
    mpi [on]                       --                                on, off                 Enable MPI support
    shared [on]                    --                                on, off                 Builds a shared version of the library
    szip [off]                     --                                on, off                 Enable szip support
    threadsafe [off]               --                                on, off                 Enable thread-safe capabilities
    tools [on]                     --                                on, off                 Enable building tools

Build Dependencies:
    cmake  gmake  java  mpi  ninja  szip  zlib

Link Dependencies:
    mpi  szip  zlib

Run Dependencies:
    java  pkgconfig
```

The spack command `spack info <package>` provides information regarding the different variants, i.e., possible configurations, and all dependencies. You can then choose to install the default version or customize your installation.

Let's check what the default configuration will install:

```bash
# show what would be installed, given a spec
(node)$ spack spec -Il hdf5
Input spec
--------------------------------
 -   hdf5

Concretized
--------------------------------
 -   be7kscq  hdf5@1.14.0%gcc@8.5.0~cxx~fortran~hl~ipo~java~map+mpi+shared~szip~threadsafe+tools api=default build_system=cmake build_type=RelWithDebInfo generator=make patches=0b5dd6f arch=linux-rhel8-zen
 -   hiorsyv      ^cmake@3.26.3%gcc@8.5.0~doc+ncurses+ownlibs~qt build_system=generic build_type=Release arch=linux-rhel8-zen
 -   3qm6oyl          ^ncurses@6.4%gcc@8.5.0~symlinks+termlib abi=none build_system=autotools arch=linux-rhel8-zen
 -   mwndgpr          ^openssl@1.1.1t%gcc@8.5.0~docs~shared build_system=generic certs=mozilla arch=linux-rhel8-zen
 -   fhqdfoi              ^ca-certificates-mozilla@2023-01-10%gcc@8.5.0 build_system=generic arch=linux-rhel8-zen
 -   nin2wpc      ^gmake@4.4.1%gcc@8.5.0~guile build_system=autotools arch=linux-rhel8-zen
 -   n2mjwkw      ^openmpi@4.1.5%gcc@8.5.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java~legacylaunchers~lustre~memchecker~orterunprefix+romio+rsh~singularity+static+vt+wrapper-rpath build_system=autotools fabrics=none schedulers=none arch=linux-rhel8-zen
 -   lqrfa5n          ^hwloc@2.9.1%gcc@8.5.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml~oneapi-level-zero~opencl+pci~rocm build_system=autotools libs=shared,static arch=linux-rhel8-zen
 -   6w6lkzs              ^libpciaccess@0.17%gcc@8.5.0 build_system=autotools arch=linux-rhel8-zen
 -   cnzqq23                  ^util-macros@1.19.3%gcc@8.5.0 build_system=autotools arch=linux-rhel8-zen
 -   xylxzqj              ^libxml2@2.10.3%gcc@8.5.0~python build_system=autotools arch=linux-rhel8-zen
 -   wv3hjx4                  ^libiconv@1.17%gcc@8.5.0 build_system=autotools libs=shared,static arch=linux-rhel8-zen
 -   r2g2ajd                  ^xz@5.4.1%gcc@8.5.0~pic build_system=autotools libs=shared,static arch=linux-rhel8-zen
 -   jeic37n          ^numactl@2.0.14%gcc@8.5.0 build_system=autotools patches=4e1d78c,62fc8a8,ff37630 arch=linux-rhel8-zen
 -   wbv247s              ^autoconf@2.69%gcc@8.5.0 build_system=autotools patches=35c4492,7793209,a49dd5b arch=linux-rhel8-zen
 -   t7feejp              ^automake@1.16.5%gcc@8.5.0 build_system=autotools arch=linux-rhel8-zen
 -   je4mbwr              ^libtool@2.4.7%gcc@8.5.0 build_system=autotools arch=linux-rhel8-zen
 -   5lv5lpk              ^m4@1.4.19%gcc@8.5.0+sigsegv build_system=autotools patches=9dc5fbd,bfdffa7 arch=linux-rhel8-zen
 -   eoimqmc                  ^diffutils@3.9%gcc@8.5.0 build_system=autotools arch=linux-rhel8-zen
 -   7yewilg                  ^libsigsegv@2.14%gcc@8.5.0 build_system=autotools arch=linux-rhel8-zen
 -   ixliisx          ^openssh@9.3p1%gcc@8.5.0+gssapi build_system=autotools arch=linux-rhel8-zen
 -   ct754qa              ^krb5@1.20.1%gcc@8.5.0+shared build_system=autotools arch=linux-rhel8-zen
 -   5vjavy3                  ^bison@3.8.2%gcc@8.5.0 build_system=autotools arch=linux-rhel8-zen
 -   sllixs3                  ^gettext@0.21.1%gcc@8.5.0+bzip2+curses+git~libunistring+libxml2+tar+xz build_system=autotools arch=linux-rhel8-zen
 -   5pqwl7d                      ^tar@1.34%gcc@8.5.0 build_system=autotools zip=pigz arch=linux-rhel8-zen
 -   4eucki5                          ^pigz@2.7%gcc@8.5.0 build_system=makefile arch=linux-rhel8-zen
 -   jnerfu4                          ^zstd@1.5.5%gcc@8.5.0+programs build_system=makefile compression=none libs=shared,static arch=linux-rhel8-zen
 -   vg5yhf2              ^libedit@3.1-20210216%gcc@8.5.0 build_system=autotools arch=linux-rhel8-zen
 -   qll3x2r              ^libxcrypt@4.4.33%gcc@8.5.0~obsolete_api build_system=autotools arch=linux-rhel8-zen
 -   y5vo5t2          ^perl@5.36.0%gcc@8.5.0+cpanm+open+shared+threads build_system=generic arch=linux-rhel8-zen
 -   uw5w4yh              ^berkeley-db@18.1.40%gcc@8.5.0+cxx~docs+stl build_system=autotools patches=26090f4,b231fcc arch=linux-rhel8-zen
 -   ymcs7ce              ^bzip2@1.0.8%gcc@8.5.0~debug~pic+shared build_system=generic arch=linux-rhel8-zen
 -   seklrqi              ^gdbm@1.23%gcc@8.5.0 build_system=autotools arch=linux-rhel8-zen
 -   zgvmdyi                  ^readline@8.2%gcc@8.5.0 build_system=autotools patches=bbf97f1 arch=linux-rhel8-zen
 -   2jwl6bm          ^pmix@4.2.3%gcc@8.5.0~docs+pmi_backwards_compatibility~python~restful build_system=autotools arch=linux-rhel8-zen
 -   bomqowi              ^libevent@2.1.12%gcc@8.5.0+openssl build_system=autotools arch=linux-rhel8-zen
 -   vr36zlh      ^pkgconf@1.8.0%gcc@8.5.0 build_system=autotools arch=linux-rhel8-zen
 -   426hs7t      ^zlib@1.2.13%gcc@8.5.0+optimize+pic+shared build_system=makefile arch=linux-rhel8-zen
```

Spack builds an installation tree:
* package already existing will have a [+] in the first column
* missing packages but installed upstream will have [^]
* unknown packages not found by Spack should be provided as external to Spack

Since we start we a fresh installation, we will need to install everything, i.e, all packages are marked as [-]. 
The ouput of the `spec` can be quite complex to read but is really exhaustive. You see all compilations options as well as the architecture and the compiler versions. There is no notion of toolchains (see for [Easybuild](http://easybuild.readthedocs.io/)).
Spack automatically detects the architecture. You can explicitely obtain it with the command `spack arch`, i.e., on Aion you will get linux-rhel8-zen.

Spack will also rely the system gcc compiler, i.e., gcc-8.5.0, to install all packages. You may want to have a newer compiler. Let's gcc 12.2.0 ...
In order to list all compilers seen by Spack, just use the command `spack compilers`.

```bash
(node)$ spack compilers
==> Available compilers
-- gcc rhel8-x86_64 ---------------------------------------------
gcc@8.5.0
```

Let's check first which gcc versions are available ...

```bash
(node)$ spack info gcc
AutotoolsPackage:   gcc

Description:
    The GNU Compiler Collection includes front ends for C, C++, Objective-C,
    Fortran, Ada, and Go, as well as libraries for these languages.

Homepage: https://gcc.gnu.org

Preferred version:  
    13.1.0    https://ftpmirror.gnu.org/gcc/gcc-13.1.0/gcc-13.1.0.tar.xz

Safe versions:  
    master    [git] git://gcc.gnu.org/git/gcc.git on branch master
    13.1.0    https://ftpmirror.gnu.org/gcc/gcc-13.1.0/gcc-13.1.0.tar.xz
    12.2.0    https://ftpmirror.gnu.org/gcc/gcc-12.2.0/gcc-12.2.0.tar.xz
    12.1.0    https://ftpmirror.gnu.org/gcc/gcc-12.1.0/gcc-12.1.0.tar.xz
    11.3.0    https://ftpmirror.gnu.org/gcc/gcc-11.3.0/gcc-11.3.0.tar.xz
    11.2.0    https://ftpmirror.gnu.org/gcc/gcc-11.2.0/gcc-11.2.0.tar.xz
    11.1.0    https://ftpmirror.gnu.org/gcc/gcc-11.1.0/gcc-11.1.0.tar.xz
    10.4.0    https://ftpmirror.gnu.org/gcc/gcc-10.4.0/gcc-10.4.0.tar.xz
    10.3.0    https://ftpmirror.gnu.org/gcc/gcc-10.3.0/gcc-10.3.0.tar.xz
    10.2.0    https://ftpmirror.gnu.org/gcc/gcc-10.2.0/gcc-10.2.0.tar.xz
    10.1.0    https://ftpmirror.gnu.org/gcc/gcc-10.1.0/gcc-10.1.0.tar.xz
    9.5.0     https://ftpmirror.gnu.org/gcc/gcc-9.5.0/gcc-9.5.0.tar.xz
    9.4.0     https://ftpmirror.gnu.org/gcc/gcc-9.4.0/gcc-9.4.0.tar.xz
    9.3.0     https://ftpmirror.gnu.org/gcc/gcc-9.3.0/gcc-9.3.0.tar.xz
    9.2.0     https://ftpmirror.gnu.org/gcc/gcc-9.2.0/gcc-9.2.0.tar.xz
    9.1.0     https://ftpmirror.gnu.org/gcc/gcc-9.1.0/gcc-9.1.0.tar.xz
    8.5.0     https://ftpmirror.gnu.org/gcc/gcc-8.5.0/gcc-8.5.0.tar.xz
    8.4.0     https://ftpmirror.gnu.org/gcc/gcc-8.4.0/gcc-8.4.0.tar.xz
    8.3.0     https://ftpmirror.gnu.org/gcc/gcc-8.3.0/gcc-8.3.0.tar.xz
    8.2.0     https://ftpmirror.gnu.org/gcc/gcc-8.2.0/gcc-8.2.0.tar.xz
    8.1.0     https://ftpmirror.gnu.org/gcc/gcc-8.1.0/gcc-8.1.0.tar.xz
    7.5.0     https://ftpmirror.gnu.org/gcc/gcc-7.5.0/gcc-7.5.0.tar.xz
    7.4.0     https://ftpmirror.gnu.org/gcc/gcc-7.4.0/gcc-7.4.0.tar.xz
    7.3.0     https://ftpmirror.gnu.org/gcc/gcc-7.3.0/gcc-7.3.0.tar.xz
    7.2.0     https://ftpmirror.gnu.org/gcc/gcc-7.2.0/gcc-7.2.0.tar.xz
    7.1.0     https://ftpmirror.gnu.org/gcc/gcc-7.1.0/gcc-7.1.0.tar.bz2
[...]
```

As expected, gcc-12.2.0 can be install through Spack.

```bash
# On Aion, you will be able to speedup compilation with the `-j128` option
(node)$ spack install -j128 gcc@12.2.0
(node)$ spack find -vpl gcc
-- linux-rhel8-zen / gcc@8.5.0 ----------------------------------
e77dbi5 gcc@12.2.0+binutils+bootstrap~graphite~nvptx~piclibs~profiled~strip build_system=autotools build_type=RelWithDebInfo languages=c,c++,fortran  /mnt/irisgpfs/users/ekieffer/.spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/gcc-12.2.0-e77dbi53lyselilz6paitdkv7osdrpfy
==> 1 installed package
```
Notice the symbol `@` to specify the version you are expecting to be installed.

Now, we need to tell Spack that this package is a compiler which can be used to build new packages

```bash
(node)$ spack compiler add "$(spack location -i gcc@12.2.0)"
==> Added 1 new compiler to /home/users/ekieffer/.spack/linux/compilers.yaml
    gcc@12.2.0
==> Compilers are defined in the following files:
    /home/users/ekieffer/.spack/linux/compilers.yaml
```

Now, we are able to built `hdf5` with gcc-12.2.0

```bash
(node)$ spack install -j128 hdf5 %gcc@12.2.0
```

Spack rely on a Domain-Specific Language (DSL) for describing the build parameters and dependencies with which a package was, or should be, built. For more informartion, plase have a look at the [Spack Specs and Dependencies documentation](https://spack.readthedocs.io/en/latest/basic_usage.html#specs-dependencies). The example below provided by the [NERSC documentation](https://docs.nersc.gov/development/build-tools/spack/) is good illustration of this DSL.

![NERSC](https://docs.nersc.gov/development/build-tools/images/spack-spec.png)



* __Package versions__:

     -  **@version** : defines a single version
     -  **@min:max** : defines version range (inclusive)
     -  **@min:**    : defines version min or higher
     -  **@:max**    : defines up to version max (inclusive)

</br>

* __Compilers__:
     - **%compiler**         : defines the compiler
     - **%compiler@version** : defines the compiler version
     - **%compiler@min:max** : defines the version range for the compiler

</br>

* __Compiler flags__:
     - **cflags="flags"** :         cppflags, cflags, cxxflags,
                                    fflags, ldflags, ldlibs
     - **==**             :         propagate flags to package dependencies

</br>

* __Variants__:
     - **+variant**                     :  enable a specific variant
     - **-variant or ~variant**         :  disable a specific variant
     - **variant=value**                :  set non-boolean variant to value
     - **variant=value1,value2,value3** :  set multi-value variant values
     - **++, --, ~~, ==**               :  propagate variants to package dependencies

</br>

* __Architecture variants__:
     - **platform=platform**        : linux, darwin, cray, etc.
     - **os=operating_system**      : specific operating system
     - **target=target**            : specific target processor
     - **arch=platform-os-target**  : shortcut for all three above

</br>

* __Dependencies__:
     - **^dependency [constraints]** : specify constraints on dependencies
     - **^/hash**                    : build with a specific installed dependency


Now, let's try to built hdf5 with mvapich2 as mpi dependencies. By default, hdf5 built openmpi.

```bash
(node)$ spack install hdf5+hl+mpi ^mvapich2
```

Check now the difference between the two installed version.

```bash
(node)$ spack find -vld  hdf5
-- linux-rhel8-zen2 / gcc@12.2.0 --------------------------------
3siutgo hdf5@1.14.0~cxx~fortran~hl~ipo~java~map+mpi+shared~szip~threadsafe+tools api=default build_system=cmake build_type=RelWithDebInfo generator=make patches=0b5dd6f
6ctbjak     cmake@3.26.3~doc+ncurses+ownlibs~qt build_system=generic build_type=Release
tzovucj         ncurses@6.4~symlinks+termlib abi=none build_system=autotools
li4zb27         openssl@1.1.1t~docs~shared build_system=generic certs=mozilla
sswrjht             ca-certificates-mozilla@2023-01-10 build_system=generic
jpezcpe     gmake@4.4.1~guile build_system=autotools
pxe6qhc     openmpi@4.1.5~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java~legacylaunchers~lustre~memchecker~orterunprefix+romio+rsh~singularity+static+vt+wrapper-rpath build_system=autotools fabrics=none schedulers=none
p45xi47         hwloc@2.9.1~cairo~cuda~gl~libudev+libxml2~netloc~nvml~oneapi-level-zero~opencl+pci~rocm build_system=autotools libs=shared,static
dx6tnq3             libpciaccess@0.17 build_system=autotools
ntnwias                 util-macros@1.19.3 build_system=autotools
porg3lf             libxml2@2.10.3~python build_system=autotools
zd5q5lz                 libiconv@1.17 build_system=autotools libs=shared,static
p35gqye                 xz@5.4.1~pic build_system=autotools libs=shared,static
553vtif         numactl@2.0.14 build_system=autotools patches=4e1d78c,62fc8a8,ff37630
l4b4gbz             autoconf@2.69 build_system=autotools patches=35c4492,7793209,a49dd5b
vi3mzkl             automake@1.16.5 build_system=autotools
m7rnule             libtool@2.4.7 build_system=autotools
7zr3ffl             m4@1.4.19+sigsegv build_system=autotools patches=9dc5fbd,bfdffa7
s44lf7l                 diffutils@3.9 build_system=autotools
ptpchf4                 libsigsegv@2.14 build_system=autotools
lgvzskp         openssh@9.3p1+gssapi build_system=autotools
qd626oc             krb5@1.20.1+shared build_system=autotools
grlxl6k                 bison@3.8.2 build_system=autotools
fwasmmb                 gettext@0.21.1+bzip2+curses+git~libunistring+libxml2+tar+xz build_system=autotools
zi3agmq                     tar@1.34 build_system=autotools zip=pigz
lje7caa                         pigz@2.7 build_system=makefile
4mz3drs                         zstd@1.5.5+programs build_system=makefile compression=none libs=shared,static
tyly5bo             libedit@3.1-20210216 build_system=autotools
yz2pcts             libxcrypt@4.4.33~obsolete_api build_system=autotools
wntkobh         perl@5.36.0+cpanm+open+shared+threads build_system=generic
nhoc6pi             berkeley-db@18.1.40+cxx~docs+stl build_system=autotools patches=26090f4,b231fcc
ms3w5jd             bzip2@1.0.8~debug~pic+shared build_system=generic
bd7mntw             gdbm@1.23 build_system=autotools
uz662wp                 readline@8.2 build_system=autotools patches=bbf97f1
rdll4qk         pmix@4.2.3~docs+pmi_backwards_compatibility~python~restful build_system=autotools
wed77jw             libevent@2.1.12+openssl build_system=autotools
q2t65hw     pkgconf@1.8.0 build_system=autotools
mou7cd5     zlib@1.2.13+optimize+pic+shared build_system=makefile

oxayojp hdf5@1.14.0~cxx~fortran+hl~ipo~java~map+mpi+shared~szip~threadsafe+tools api=default build_system=cmake build_type=RelWithDebInfo generator=make patches=0b5dd6f
6ctbjak     cmake@3.26.3~doc+ncurses+ownlibs~qt build_system=generic build_type=Release
tzovucj         ncurses@6.4~symlinks+termlib abi=none build_system=autotools
li4zb27         openssl@1.1.1t~docs~shared build_system=generic certs=mozilla
sswrjht             ca-certificates-mozilla@2023-01-10 build_system=generic
wntkobh             perl@5.36.0+cpanm+open+shared+threads build_system=generic
nhoc6pi                 berkeley-db@18.1.40+cxx~docs+stl build_system=autotools patches=26090f4,b231fcc
ms3w5jd                 bzip2@1.0.8~debug~pic+shared build_system=generic
bd7mntw                 gdbm@1.23 build_system=autotools
uz662wp                     readline@8.2 build_system=autotools patches=bbf97f1
jpezcpe     gmake@4.4.1~guile build_system=autotools
fnh3qls     mvapich2@2.3.7~alloca~cuda~debug~hwlocv2+regcache+wrapperrpath build_system=autotools ch3_rank_bits=32 fabrics=mrail file_systems=auto process_managers=auto threads=multiple
grlxl6k         bison@3.8.2 build_system=autotools
s44lf7l             diffutils@3.9 build_system=autotools
7zr3ffl             m4@1.4.19+sigsegv build_system=autotools patches=9dc5fbd,bfdffa7
ptpchf4                 libsigsegv@2.14 build_system=autotools
3nkcsyz         findutils@4.9.0 build_system=autotools patches=440b954
dx6tnq3         libpciaccess@0.17 build_system=autotools
m7rnule             libtool@2.4.7 build_system=autotools
ntnwias             util-macros@1.19.3 build_system=autotools
porg3lf         libxml2@2.10.3~python build_system=autotools
zd5q5lz             libiconv@1.17 build_system=autotools libs=shared,static
p35gqye             xz@5.4.1~pic build_system=autotools libs=shared,static
6wpwqr5         rdma-core@41.0~ipo+static build_system=cmake build_type=RelWithDebInfo generator=make
kezio66             libnl@3.3.0 build_system=autotools
mnpw6rx                 flex@2.6.3+lex~nls build_system=autotools
cqa2b2a             py-docutils@0.19 build_system=python_pip
ef5nrmp                 py-pip@23.0 build_system=generic
2ajcrnb                 py-setuptools@67.6.0 build_system=generic
7mvi3qw                 py-wheel@0.37.1 build_system=generic
z7gsvzc                 python@3.10.10+bz2+crypt+ctypes+dbm~debug+libxml2+lzma~nis~optimizations+pic+pyexpat+pythoncmd+readline+shared+sqlite3+ssl~tkinter+uuid+zlib build_system=generic patches=0d98e93,7d40923,f2fd060
vnnaacu                     expat@2.5.0+libbsd build_system=autotools
ffgqjcw                         libbsd@0.11.7 build_system=autotools
27arg6b                             libmd@1.0.4 build_system=autotools
fwasmmb                     gettext@0.21.1+bzip2+curses+git~libunistring+libxml2+tar+xz build_system=autotools
zi3agmq                         tar@1.34 build_system=autotools zip=pigz
lje7caa                             pigz@2.7 build_system=makefile
4mz3drs                             zstd@1.5.5+programs build_system=makefile compression=none libs=shared,static
hmewqfx                     libffi@3.4.4 build_system=autotools
yz2pcts                     libxcrypt@4.4.33~obsolete_api build_system=autotools
jx26hkw                     sqlite@3.40.1+column_metadata+dynamic_extensions+fts~functions+rtree build_system=autotools
kr76dbm                     util-linux-uuid@2.38.1 build_system=autotools
q2t65hw     pkgconf@1.8.0 build_system=autotools
mou7cd5     zlib@1.2.13+optimize+pic+shared build_system=makefile
```

The installation is not destructive. Each version has its own hash value displayed on the first column.


### Part 2: Interaction with Schedulers (slurm)

[Slurm](https://slurm.schedmd.com/) is the ULHPC supported batch scheduler, this package should **never** be installed by spack. The ULHPC's slurm has been built with pmix and both tools should be external dependencies for spack. 

Why is it so important ?

If you don't specify to Spack the scheduler, you will **NOT** benefit from the slurm support to start distributed jobs. You will need to use the legacy approach, i.e., mpirun -np ${SLURM_NTASKS} --hostfile hosts.

In order to add slurm and pmix as externals dependencies, please use the follow bash script.

```bash
cat << EOF >> $SPACK_ROOT/etc/spack/packages.yaml
packages:
    slurm:
        externals:
        - spec: slurm@22.05.5
          prefix: /usr
        buildable: False
    libevent:
        externals:
        - spec: libevent@2.1.8
          prefix: /usr
        buildable: False
    pmix:
        externals:
        - spec: pmix@4.2.3 
          prefix: /usr
        buildable: False
    hwloc:
        externals:
        - spec: hwloc@2.2.0
          prefix: /usr
        buildable: False
EOF
```

Now, let's try to install openmpi with slurm support.

```bash
(node)$ spack install -j128 openmpi@4.0.5 +pmi schedulers=slurm  ^pmix@4.2.3 ^hwloc@2.2.0
(node)$ spack find -vpl openmpi
```
Let's try if the slurm support works properly.


```bash
cat << EOF >> launcher_osu.sh
#!/bin/bash -l
#SBATCH --job-name=mpi_job_test      # Job name
#SBATCH --cpus-per-task=1            # Number of cores per MPI task
#SBATCH --nodes=2                    # Maximum number of nodes to be allocated
#SBATCH --ntasks-per-node=128        # Maximum number of tasks on each node
#SBATCH --output=mpi_test_%j.log     # Path to the standard output and error files relative to the working directory
#SBATCH --exclusive
#SBATCH -p batch

export SRUN_CPUS_PER_TASK=\${SLURM_CPUS_PER_TASK}
OSU_VERSION="7.1-1"
OSU_ARCHIVE="osu-micro-benchmarks-\${OSU_VERSION}.tar.gz"
OSU_URL="https://mvapich.cse.ohio-state.edu/download/mvapich/\${OSU_ARCHIVE}"

if [[ ! -f \${OSU_ARCHIVE} ]];then 
    wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.1-1.tar.gz
    tar -xvf \${OSU_ARCHIVE} 
fi

# We use the hash since we could have different variants of the same openmpi version
# Adapt with your hash version
spack load /xgcbqft

# cd into the extracted folder 
cd \${OSU_ARCHIVE//.tar.gz/}

# configure
./configure CC=\$(which mpicc) CXX=\$(which mpicxx)
make
cd ..

srun  \${OSU_ARCHIVE//.tar.gz/}/c/mpi/collective/blocking/osu_alltoall
EOF
```

* Start the previous script with the following command `sbatch launcher_osu.sh`  

You should see something like this

```bash
[...]
# OSU MPI All-to-All Personalized Exchange Latency Test v7.1
# Datatype: MPI_CHAR.
# Size       Avg Latency(us)
1                     478.18
2                     699.57
4                    1048.96
8                    1858.79
16                   3638.70
32                   7154.35
64                  14922.24
128                 31227.27
256                 70394.74
```

That is all for now :)

## Reference

* [Spack documentation](https://spack.readthedocs.io/en/latest/index.html#)
* [Reference for the slurm support](https://gchp.readthedocs.io/en/13.1.0/supplement/spack.html)


### Spack environments

Spack environments are a feature of the Spack package manager, which is an open-source tool for managing software installations and dependencies on HPC (High-Performance Computing) systems, clusters, and other Unix-based environments. Spack environments provide a way to describe, reproduce, and manage sets of packages and their dependencies in a single, self-contained configuration.

Spack environments have the following key features:

1. Isolation: Environments create isolated spaces where packages and their dependencies can be installed without interfering with each other or the global Spack installation.

2. Reproducibility: Environments can be defined using a YAML configuration file (spack.yaml or spack.lock), which lists the desired packages and their versions. This makes it easy to share and reproduce software installations across different systems or users.

3. Flexibility: Users can create multiple environments to manage different sets of packages or software stacks for different projects or tasks.

4. Version control: The configuration files can be placed under version control, enabling tracking of changes, collaboration, and easy rollback to previous configurations if needed.

5. Integration: Spack environments can be integrated with other tools and workflows, such as continuous integration systems, containerization tools, or job schedulers in HPC environments.

In summary, Spack environments provide a convenient and flexible way to manage software installations and dependencies, enhancing productivity and collaboration in research and development, particularly in the HPC community.

If you have followed the beginning of this tutorial, you should already have some spack packages:

```bash
(node)$ spack find
-- linux-rhel8-zen / gcc@=8.5.0 ---------------------------------
autoconf@2.69                ca-certificates-mozilla@2023-01-10  krb5@1.20.1           libtool@2.4.7     openmpi@4.0.5   parallel@20220522  slurm@22.05.5       zlib@1.2.13
autoconf-archive@2023.02.20  diffutils@3.9                       libedit@3.1-20210216  libxcrypt@4.4.33  openmpi@4.0.5   perl@5.36.0        slurm@22.05.5       zstd@1.5.5
automake@1.16.5              gdbm@1.23                           libfabric@1.18.0      libxml2@2.10.3    openmpi@4.1.5   pigz@2.7           tar@1.34
berkeley-db@18.1.40          gettext@0.21.1                      libiconv@1.17         m4@1.4.19         openmpi@4.1.5   pkgconf@1.8.0      ucx@1.14.0
bison@3.8.2                  gmake@4.4.1                         libpciaccess@0.17     ncurses@6.4       openssh@9.3p1   pmix@4.2.3         util-macros@1.19.3
bzip2@1.0.8                  hwloc@2.2.0                         libsigsegv@2.14       numactl@2.0.14    openssl@1.1.1t  readline@8.2       xz@5.4.1
==> 44 installed packages
```

* Let's create now an isolated environement

```bash
(node)$ spack env create myenv
(node)$ spack env list
==> 1 environments
    myenv
```

An environment is similar as a virtualized Spack instance used to collect and aggregate package installations for a specific project. The concept is very close to virtual environments in python.

The `myenv`environment can be now activated using the following spack command: `spack env activate -p myproject`. Using `-p` option will add the activated environment name to your prompt.

* You can check in which environment you are located using `spack env status`
* If you try now `spack find`, you should not see any installed packages. 

Let's try now to install some packages...

```bash
(node)$ spack install zlib
==> Error: Cannot install 'zlib' because no matching specs are in the current environment. You can add specs to the environment with 'spack add zlib', or as part of the install command with 'spack install --add zlib'
```
In order to install packages, you need first to add specs. Internally, spack will queue the specs to be installed together. Let's add other packages to our environment:

```bash
(node)$ spack add tcl 
==> Adding tcl to environment myenv
(node)$ spack add eigen
==> Adding eigen to environment myenv
# Now we can install the packages
(node)$ spack install
==> Concretized tcl
 -   l5ue6gx  tcl@=8.6.12%gcc@=8.5.0 build_system=autotools arch=linux-rhel8-zen
[+]  426hs7t      ^zlib@=1.2.13%gcc@=8.5.0+optimize+pic+shared build_system=makefile arch=linux-rhel8-zen

==> Concretized eigen
 -   g3mjusn  eigen@=3.4.0%gcc@=8.5.0~ipo build_system=cmake build_type=RelWithDebInfo generator=make arch=linux-rhel8-zen
 -   3zmanxo      ^cmake@=3.26.3%gcc@=8.5.0~doc+ncurses+ownlibs~qt build_system=generic build_type=Release arch=linux-rhel8-zen
[+]  3qm6oyl          ^ncurses@=6.4%gcc@=8.5.0~symlinks+termlib abi=none build_system=autotools arch=linux-rhel8-zen
[+]  vr36zlh              ^pkgconf@=1.8.0%gcc@=8.5.0 build_system=autotools arch=linux-rhel8-zen
[+]  fd7rnqx          ^openssl@=1.1.1t%gcc@=8.5.0~docs~shared build_system=generic certs=mozilla arch=linux-rhel8-zen
[+]  fhqdfoi              ^ca-certificates-mozilla@=2023-01-10%gcc@=8.5.0 build_system=generic arch=linux-rhel8-zen
[+]  yvgdvqo              ^perl@=5.36.0%gcc@=8.5.0+cpanm+open+shared+threads build_system=generic arch=linux-rhel8-zen
[+]  uw5w4yh                  ^berkeley-db@=18.1.40%gcc@=8.5.0+cxx~docs+stl build_system=autotools patches=26090f4,b231fcc arch=linux-rhel8-zen
[+]  ymcs7ce                  ^bzip2@=1.0.8%gcc@=8.5.0~debug~pic+shared build_system=generic arch=linux-rhel8-zen
[+]  eoimqmc                      ^diffutils@=3.9%gcc@=8.5.0 build_system=autotools arch=linux-rhel8-zen
[+]  wv3hjx4                          ^libiconv@=1.17%gcc@=8.5.0 build_system=autotools libs=shared,static arch=linux-rhel8-zen
[+]  seklrqi                  ^gdbm@=1.23%gcc@=8.5.0 build_system=autotools arch=linux-rhel8-zen
[+]  zgvmdyi                      ^readline@=8.2%gcc@=8.5.0 build_system=autotools patches=bbf97f1 arch=linux-rhel8-zen
[+]  426hs7t              ^zlib@=1.2.13%gcc@=8.5.0+optimize+pic+shared build_system=makefile arch=linux-rhel8-zen
[+]  nin2wpc      ^gmake@=4.4.1%gcc@=8.5.0~guile build_system=autotools arch=linux-rhel8-zen

[+] /mnt/irisgpfs/users/ekieffer/.spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/zlib-1.2.13-426hs7tsxcfpebed5uqlogma32dbuvj5
[+] /mnt/irisgpfs/users/ekieffer/.spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/ncurses-6.4-3qm6oylywjcvizw7xyqbkxg33vqtgppp
[+] /mnt/irisgpfs/users/ekieffer/.spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/gmake-4.4.1-nin2wpcxfbi5jfu56hh2zah4giapsll7
==> Installing tcl-8.6.12-l5ue6gxjdrpsoix4fdn6sgcl3bma6qaj
==> No binary for tcl-8.6.12-l5ue6gxjdrpsoix4fdn6sgcl3bma6qaj found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/26/26c995dd0f167e48b11961d891ee555f680c175f7173ff8cb829f4ebcde4c1a6.tar.gz
==> No patches needed for tcl
==> tcl: Executing phase: 'autoreconf'
==> tcl: Executing phase: 'configure'
==> tcl: Executing phase: 'build'
==> tcl: Executing phase: 'install'
==> tcl: Successfully installed tcl-8.6.12-l5ue6gxjdrpsoix4fdn6sgcl3bma6qaj
  Stage: 1.38s.  Autoreconf: 0.00s.  Configure: 5.56s.  Build: 46.11s.  Install: 4.96s.  Total: 58.11s
[+] /mnt/irisgpfs/users/ekieffer/.spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/tcl-8.6.12-l5ue6gxjdrpsoix4fdn6sgcl3bma6qaj
[+] /mnt/irisgpfs/users/ekieffer/.spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/openssl-1.1.1t-fd7rnqxs5w7u6hdple4jdxobefeceevs
==> Installing cmake-3.26.3-3zmanxoso62q3bnzejrlnmpna4gas4bk
==> No binary for cmake-3.26.3-3zmanxoso62q3bnzejrlnmpna4gas4bk found: installing from source
==> Fetching https://github.com/Kitware/CMake/releases/download/v3.26.3/cmake-3.26.3.tar.gz
==> No patches needed for cmake
==> cmake: Executing phase: 'bootstrap'
==> cmake: Executing phase: 'build'
==> cmake: Executing phase: 'install'
==> cmake: Successfully installed cmake-3.26.3-3zmanxoso62q3bnzejrlnmpna4gas4bk
  Stage: 3.11s.  Bootstrap: 55.36s.  Build: 56.02s.  Install: 4.00s.  Total: 1m 58.79s
[+] /mnt/irisgpfs/users/ekieffer/.spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/cmake-3.26.3-3zmanxoso62q3bnzejrlnmpna4gas4bk
==> Installing eigen-3.4.0-g3mjusnpkiwnt2xu4fhhv4szbocwdfym
==> No binary for eigen-3.4.0-g3mjusnpkiwnt2xu4fhhv4szbocwdfym found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/85/8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72.tar.gz
==> No patches needed for eigen
==> eigen: Executing phase: 'cmake'
==> eigen: Executing phase: 'build'
==> eigen: Executing phase: 'install'
==> eigen: Successfully installed eigen-3.4.0-g3mjusnpkiwnt2xu4fhhv4szbocwdfym
  Stage: 1.09s.  Cmake: 6.13s.  Build: 0.23s.  Install: 0.85s.  Total: 8.44s
[+] /mnt/irisgpfs/users/ekieffer/.spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/eigen-3.4.0-g3mjusnpkiwnt2xu4fhhv4szbocwdfym
==> Updating view at /mnt/irisgpfs/users/ekieffer/.spack/var/spack/environments/myenv/.spack-env/view
```

When you activate a spack environment, all packages are automatically loaded and accessible. You do not need to call `spack load`.

"**When you install packages into an environment, they are, by default, linked into a single prefix, or view. Activating the environment with spack env activate results in subdirectories from the view being added to PATH, MANPATH, CMAKE_PREFIX_PATH, and other environment variables. This makes the environment easier to use.**"

```bash
(node)$ which tclsh
/mnt/irisgpfs/users/ekieffer/.spack/var/spack/environments/myenv/.spack-env/view/bin/tclsh
```

* In order to remove a package, use `spack uninstall --remove <package>`.

* The contents of environments is tracked by two files: 
  - `spack.yaml`: holds the environment configuration (abstract specs to install)
  - `spack.lock`: generated during concretization (full concrete specs)

* You can use both files to transfer your software environement to another one.

```bash
# Copy configuration from the myenv environment
(node)$ cp ${SPACK_ROOT}/var/spack/environments/myenv/spack.yaml .
# Create a new environment from the configuration file
(node)$ spack env create myenv2 spack.yaml
(node)$ spack env activate myenv2
# Installing software ... What do you observe ?
(node)$ spack install
==> Concretized tcl
[+]  l5ue6gx  tcl@=8.6.12%gcc@=8.5.0 build_system=autotools arch=linux-rhel8-zen
[+]  426hs7t      ^zlib@=1.2.13%gcc@=8.5.0+optimize+pic+shared build_system=makefile arch=linux-rhel8-zen

==> Concretized eigen
[+]  g3mjusn  eigen@=3.4.0%gcc@=8.5.0~ipo build_system=cmake build_type=RelWithDebInfo generator=make arch=linux-rhel8-zen
[+]  3zmanxo      ^cmake@=3.26.3%gcc@=8.5.0~doc+ncurses+ownlibs~qt build_system=generic build_type=Release arch=linux-rhel8-zen
[+]  3qm6oyl          ^ncurses@=6.4%gcc@=8.5.0~symlinks+termlib abi=none build_system=autotools arch=linux-rhel8-zen
[+]  vr36zlh              ^pkgconf@=1.8.0%gcc@=8.5.0 build_system=autotools arch=linux-rhel8-zen
[+]  fd7rnqx          ^openssl@=1.1.1t%gcc@=8.5.0~docs~shared build_system=generic certs=mozilla arch=linux-rhel8-zen
[+]  fhqdfoi              ^ca-certificates-mozilla@=2023-01-10%gcc@=8.5.0 build_system=generic arch=linux-rhel8-zen
[+]  yvgdvqo              ^perl@=5.36.0%gcc@=8.5.0+cpanm+open+shared+threads build_system=generic arch=linux-rhel8-zen
[+]  uw5w4yh                  ^berkeley-db@=18.1.40%gcc@=8.5.0+cxx~docs+stl build_system=autotools patches=26090f4,b231fcc arch=linux-rhel8-zen
[+]  ymcs7ce                  ^bzip2@=1.0.8%gcc@=8.5.0~debug~pic+shared build_system=generic arch=linux-rhel8-zen
[+]  eoimqmc                      ^diffutils@=3.9%gcc@=8.5.0 build_system=autotools arch=linux-rhel8-zen
[+]  wv3hjx4                          ^libiconv@=1.17%gcc@=8.5.0 build_system=autotools libs=shared,static arch=linux-rhel8-zen
[+]  seklrqi                  ^gdbm@=1.23%gcc@=8.5.0 build_system=autotools arch=linux-rhel8-zen
[+]  zgvmdyi                      ^readline@=8.2%gcc@=8.5.0 build_system=autotools patches=bbf97f1 arch=linux-rhel8-zen
[+]  426hs7t              ^zlib@=1.2.13%gcc@=8.5.0+optimize+pic+shared build_system=makefile arch=linux-rhel8-zen
[+]  nin2wpc      ^gmake@=4.4.1%gcc@=8.5.0~guile build_system=autotools arch=linux-rhel8-zen

==> All of the packages are already installed
==> Updating view at /mnt/irisgpfs/users/ekieffer/.spack/var/spack/environments/myenv2/.spack-env/view

# To deactivate
(node)$ spack env deactivate
```
* We can modify packages from an environment without affecting other environments. Environments only link to those existing installations.

For more details on Spack, please refer to the [official documentation](https://spack.readthedocs.io/en/latest/). 














