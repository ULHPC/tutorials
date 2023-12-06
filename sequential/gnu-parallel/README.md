[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/sequential/basics/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/sequential/gnu-parallel/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/sequential/gnu-parallel/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Distributing embarrassingly parallel tasks GNU Parallel

     Copyright (c) 2021 UL HPC Team <hpc-team@uni.lu>

![](https://www.gnu.org/software/parallel/logo-gray+black300.png)

[GNU Parallel](http://www.gnu.org/software/parallel/) is a tool for executing tasks in parallel, typically on a single machine. **When coupled with the Slurm command `srun`, parallel becomes a powerful way of distributing a set of tasks amongst a number of workers**. This is particularly useful when the number of tasks is significantly larger than the number of available workers (i.e. `$SLURM_NTASKS`), and each tasks is independent of the others.


## Prerequisites

The `parallel` command is available at the system level across the ULHPC clusters, yet under a relatively old version:

``` bash
(access)$ which parallel
/usr/bin/parallel
(access)$ parallel --version
GNU parallel 20190922
[...]
```

If you want to build a more recent version. The process is quite straight-forward and we will illustrate this process Easybuild (see the [Easybuild tutorial](../../tools/easybuild/)  and [Spack tutorial](../../tools/spack/)). 

### With Easybuild

* We are going to extend the current software set to avoid recompiling a toolchain
* The parallel version will be built using GCC-10.2.0

```bash
(node)$ resif-load-home-swset-prod
# Check where it will be installed
(node)$ echo $MODULEPATH
# Loading Easybuild
(node)$ module load tools/EasyBuild
(node)$ eb -S parallel
eb -S parallel
== found valid index for /opt/apps/resif/aion/2020b/epyc/software/EasyBuild/4.5.4/easybuild/easyconfigs, so using it...
CFGS1=/opt/apps/resif/aion/2020b/epyc/software/EasyBuild/4.5.4/easybuild/easyconfigs
 * $CFGS1/a/Amber/Amber-16-AT-17-fix_missing_do_parallel_in_checkrismunsupported.patch
 * $CFGS1/a/Amber/Amber-18-AT-18_fix_missing_do_parallel_in_checkrismunsupported.patch
 * $CFGS1/h/HPL/HPL_parallel-make.patch
 * $CFGS1/i/ipyparallel/ipyparallel-6.2.2-foss-2018a-Python-3.6.4.eb
 * $CFGS1/j/Judy/Judy-1.0.5_parallel-make.patch
 * $CFGS1/n/NWChem/NWChem-6.3.revision2-parallelbuild.patch
 * $CFGS1/n/NWChem/NWChem-6.5.revision26243-parallelbuild.patch
 * $CFGS1/n/NWChem/NWChem-6.6.revision27746-parallelbuild.patch
 * $CFGS1/n/netCDF/netCDF-4.3.2-parallel-HDF.patch
 * $CFGS1/o/OpenSSL/OpenSSL-1.0.1i-fix_parallel_build-1.patch
 * $CFGS1/o/OpenSSL/OpenSSL-1.0.1m_fix-parallel.patch
 * $CFGS1/o/OpenSees/OpenSees-3.2.0-add_Makefile_def_parallel.patch
 * $CFGS1/o/OpenSees/OpenSees-3.2.0-intel-2020a-Python-3.8.2-parallel.eb
 * $CFGS1/p/ParallelIO/ParallelIO-2.2.2a-intel-2017a.eb
 * $CFGS1/p/PyTorch/PyTorch-1.7.0_fix_test_DistributedDataParallel.patch
 * $CFGS1/p/parallel-fastq-dump/parallel-fastq-dump-0.6.5-GCCcore-8.2.0-Python-3.7.2.eb
 * $CFGS1/p/parallel-fastq-dump/parallel-fastq-dump-0.6.6-GCCcore-9.3.0-Python-3.8.2.eb
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
 * $CFGS1/r/R/DMCfun-1.3.0_fix-parallel-detect.patch
 * $CFGS1/w/WRF/WRF_parallel_build_fix.patch
 * $CFGS1/x/Xmipp/Xmipp-3.19.04-Apollo_add_missing_pthread_to_XmippParallel.patch

Note: 7 matching archived easyconfig(s) found, use --consider-archived-easyconfigs to see them
```

* Let's install `$CFGS1/p/parallel/parallel-20210322-GCCcore-10.2.0.eb`

```bash
(node)$ eb parallel-20210322-GCCcore-10.2.0.eb -r 
(node)$ module av parallel

-------------------------------------------------------------------------------------------------------------------------- /home/users/ekieffer/.local/easybuild/aion/2020b/epyc/modules/all ---------------------------------------------------------------------------------------------------------------------------
   tools/parallel/20210322-GCCcore-10.2.0

If the avail list is too long consider trying:

"module --default avail" or "ml -d av" to just list the default modules.
"module overview" or "ml ov" to display the number of modules for each name.

Use "module spider" to find all possible modules and extensions.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".
```

### With Spack

* If we don't have it installed, please follow first [the Spack tutorial](../../tools/spack/)


```bash
(node)$ spack list parallel
aws-parallelcluster  intel-parallel-studio  parallel  parallel-netcdf  parallelio  parallelmergetree  perl-parallel-forkmanager  py-ipyparallel  py-pytest-parallel  r-biocparallel  r-doparallel  r-optimparallel  r-parallelly  r-parallelmap  r-rcppparallel
==> 15 packages
(node)$ spack versions parallel
==> Safe versions (already checksummed):
  20220522  20220422  20220322  20220222  20220122  20210922  20200822  20190222  20170322  20170122  20160422  20160322
==> Remote versions (not yet checksummed):
  20230422  20221222  20220822  20211122  20210622  20210222  20201022  20200522  20200122  20190922  20190522  20181222  20180822  20180422  20171222  20170822  20170422  20161022  20160622  20151222  20150822  20150422  20141122  20140722  20140322  20131122  20130722  20130222  20121022  20120522  20120122
  20230322  20221122  20220722  20211022  20210522  20210122  20200922  20200422  20191222  20190822  20190422  20181122  20180722  20180322  20171122  20170722  20170222  20160922  20160522  20151122  20150722  20150322  20141022  20140622  20140222  20131022  20130622  20130122  20120822  20120422
  20230222  20221022  20220622  20210822  20210422  20201222  20200722  20200322  20191122  20190722  20190322  20181022  20180622  20180222  20171022  20170622  20161222  20160822  20160222  20151022  20150622  20150222  20140922  20140522  20140122  20130922  20130522  20121222  20120722  20120322
  20230122  20220922  20211222  20210722  20210322  20201122  20200622  20200222  20191022  20190622  20190122  20180922  20180522  20180122  20170922  20170522  20161122  20160722  20160122  20150922  20150522  20150122  20140822  20140422  20131222  20130822  20130422  20121122  20120622  20120222
```

* The most recent and safe version is `20220522`

```bash
(node)$ spack install parallel@20220522
[+] /mnt/irisgpfs/users/ekieffer/.spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/berkeley-db-18.1.40-uw5w4yhzzi2fatjzb72ipgdf3w657tle
[+] /mnt/irisgpfs/users/ekieffer/.spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/bzip2-1.0.8-ymcs7cevgovcd3bc5iphzo5ztzv62jue
[+] /mnt/irisgpfs/users/ekieffer/.spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/ncurses-6.4-3qm6oylywjcvizw7xyqbkxg33vqtgppp
[+] /mnt/irisgpfs/users/ekieffer/.spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/zlib-1.2.13-426hs7tsxcfpebed5uqlogma32dbuvj5
[+] /mnt/irisgpfs/users/ekieffer/.spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/readline-8.2-zgvmdyizb6g4ee2ozansvqkxvgbq6a6r
[+] /mnt/irisgpfs/users/ekieffer/.spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/gdbm-1.23-seklrqiazmh54ts3tdx6fpnlobviw3ia
[+] /mnt/irisgpfs/users/ekieffer/.spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/perl-5.36.0-yvgdvqozt46iimytfldxpjhxxao2gtdy
==> Installing parallel-20220522-ediq6fzty3il5qc3frzlwn45szfvbhek
==> No binary for parallel-20220522-ediq6fzty3il5qc3frzlwn45szfvbhek found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/bb/bb6395f8d964e68f3bdb26a764d3c48b69bc5b759a92ac3ab2bd1895c7fa8c1f.tar.bz2
==> No patches needed for parallel
==> parallel: Executing phase: 'autoreconf'
==> parallel: Executing phase: 'configure'
==> parallel: Executing phase: 'build'
==> parallel: Executing phase: 'install'
==> parallel: Successfully installed parallel-20220522-ediq6fzty3il5qc3frzlwn45szfvbhek
  Stage: 2.99s.  Autoreconf: 0.00s.  Configure: 1.20s.  Build: 0.02s.  Install: 0.60s.  Total: 4.90s
[+] /mnt/irisgpfs/users/ekieffer/.spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/parallel-20220522-ediq6fzty3il5qc3frzlwn45szfvbhek
(node)$ spack find -vpl parallel
-- linux-rhel8-zen / gcc@=8.5.0 ---------------------------------
ediq6fz parallel@20220522 build_system=autotools  /mnt/irisgpfs/users/ekieffer/.spack/opt/spack/linux-rhel8-zen/gcc-8.5.0/parallel-20220522-ediq6fzty3il5qc3frzlwn45szfvbhek
==> 1 installed package
```







## Discovering the `parallel` command

The [GNU Parallel](http://www.gnu.org/software/parallel/) syntax can be a little distributing, but basically it supports two modes:

* Reading command arguments on the command line:

        parallel	[-j N] [OPTIONS]	COMMAND	{} ::: TASKLIST

* Reading command arguments from an input file:

        parallel	–a	TASKLIST.LST	[-j N] [OPTIONS]	COMMAND	{}
        parallel	[-j N] [OPTIONS]	COMMAND	{} :::: TASKLIST.LST

If your COMMAND embed a pipe stage, you have to escape the pipe symbol as follows `\|`.
Let's make some tests. The `-j <N>` option permits to define the jobs per machine - in particular you may want to use `-j 1` to enable a sequential resolution of the parallel command.

In all cases, the `parallel` command is available at the system across the ULHPC clusters. Run it once.

``` bash
(access)$> parallel --version
GNU parallel 20160222
Copyright (C) 2007,2008,2009,2010,2011,2012,2013,2014,2015,2016
Ole Tange and Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
GNU parallel comes with no warranty.

Web site: http://www.gnu.org/software/parallel

When using programs that use GNU Parallel to process data for publication
please cite as described in 'parallel --bibtex'.
```

If you want to avoid the disclaimer requesting to cite the paper describing GNU parallel, you have to run as indicated `parallel --bibtex`, type: 'will cite' and press enter.

Let's play with a __TASKLIST from the command line__:

``` bash
(access)$> parallel echo {} ::: A B C
A
B
C
(access)$> parallel echo {} ::: {1..3}
1
2
3
# As above
(access)$> parallel echo {} ::: $(seq 1 3)
1
2
3
# Use index to refer to a given TASKLIST
(access)$> parallel echo {1} {2} ::: A B C ::: {1..3}
A 1
A 2
A 3
B 1
B 2
B 3
C 1
C 2
C 3
# Combine (link) TASKLIST with same size - Use '--link' on more recent parallel version
(access)$> parallel --xapply "echo {1} {2}" ::: A B C ::: {1..3}
A 1
B 2
C 3
# This can be useful to output command text with arguments
(access)$> parallel --xapply echo myapp_need_argument {1} {2} ::: A B C ::: {1..3}
myapp_need_argument A 1
myapp_need_argument B 2
myapp_need_argument C 3
# /!\ IMPORTANT: you can then **execute** these commands as above  by removing 'echo'
#     DON'T do that unless you know what you're doing
# You can filter out some elements:
(access)$> parallel --xapply echo myapp_need_argument {1} {2} \| grep -v 2 ::: A B C ::: {1..3}
myapp_need_argument A 1
myapp_need_argument C 3
```

Let's play now with a __TASKLIST from an input file__.

Let's assume you wish to process some images from the [OpenImages V4 data set](https://storage.googleapis.com/openimages/web/download_v4.html).
A copy of this data set is available on the ULHPC facility, under `/work/projects/bigdata_sets/OpenImages_V4/`.
Let's create a CSV file which contains a random selection of 10 training files within this dataset (prefixed by a line number).
You may want to do it as follows (**copy the full command**):

``` bash
#                                                       training set     select first 10K  random sort  take only top 10   prefix by line number      print to stdout AND in file
#                                                         ^^^^^^           ^^^^^^^^^^^^^   ^^^^^^^^     ^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(access)$> find /work/projects/bigdata_sets/OpenImages_V4/train/ -print | head -n 10000 | sort -R   |  head -n 10       | awk '{ print ++i","$0 }' | tee openimages_v4_filelist.csv
1,/work/projects/bigdata_sets/OpenImages_V4/train/6196380ea79283e0.jpg
2,/work/projects/bigdata_sets/OpenImages_V4/train/7f23f40740731c03.jpg
3,/work/projects/bigdata_sets/OpenImages_V4/train/dbfc1b37f45b3957.jpg
4,/work/projects/bigdata_sets/OpenImages_V4/train/f66087cdf8e172cd.jpg
5,/work/projects/bigdata_sets/OpenImages_V4/train/5efed414dd8b23d0.jpg
6,/work/projects/bigdata_sets/OpenImages_V4/train/1be054cb3021f6aa.jpg
7,/work/projects/bigdata_sets/OpenImages_V4/train/61446dee2ee9eb27.jpg
8,/work/projects/bigdata_sets/OpenImages_V4/train/dba2da75d899c3e7.jpg
9,/work/projects/bigdata_sets/OpenImages_V4/train/7ea06f092abc005e.jpg
10,/work/projects/bigdata_sets/OpenImages_V4/train/2db694eba4d4bb04.jpg
```

Let's manipulate the file content with parallel (prefer the `-a <filename>` syntax):

``` bash
# simple echo of the file
(access)$> parallel -a openimages_v4_filelist.csv echo {}
1,/work/projects/bigdata_sets/OpenImages_V4/train/6196380ea79283e0.jpg
2,/work/projects/bigdata_sets/OpenImages_V4/train/7f23f40740731c03.jpg
3,/work/projects/bigdata_sets/OpenImages_V4/train/dbfc1b37f45b3957.jpg
[...]
# print specific column of the CSV file
(access)$> parallel --colsep '\,' -a openimages_v4_filelist.csv echo {1}
1
2
3
4
5
6
7
8
9
10
(access)$> parallel --colsep '\,' -a openimages_v4_filelist.csv echo {2}
/work/projects/bigdata_sets/OpenImages_V4/train/6196380ea79283e0.jpg
/work/projects/bigdata_sets/OpenImages_V4/train/7f23f40740731c03.jpg
/work/projects/bigdata_sets/OpenImages_V4/train/dbfc1b37f45b3957.jpg
/work/projects/bigdata_sets/OpenImages_V4/train/f66087cdf8e172cd.jpg
/work/projects/bigdata_sets/OpenImages_V4/train/5efed414dd8b23d0.jpg
/work/projects/bigdata_sets/OpenImages_V4/train/1be054cb3021f6aa.jpg
/work/projects/bigdata_sets/OpenImages_V4/train/61446dee2ee9eb27.jpg
/work/projects/bigdata_sets/OpenImages_V4/train/dba2da75d899c3e7.jpg
/work/projects/bigdata_sets/OpenImages_V4/train/7ea06f092abc005e.jpg
/work/projects/bigdata_sets/OpenImages_V4/train/2db694eba4d4bb04.jpg

# reformat and change order
(access)$> parallel --colsep '\,' -a openimages_v4_filelist.csv echo {2} {1}
/work/projects/bigdata_sets/OpenImages_V4/train/6196380ea79283e0.jpg 1
/work/projects/bigdata_sets/OpenImages_V4/train/7f23f40740731c03.jpg 2
/work/projects/bigdata_sets/OpenImages_V4/train/dbfc1b37f45b3957.jpg 3
/work/projects/bigdata_sets/OpenImages_V4/train/f66087cdf8e172cd.jpg 4
/work/projects/bigdata_sets/OpenImages_V4/train/5efed414dd8b23d0.jpg 5
/work/projects/bigdata_sets/OpenImages_V4/train/1be054cb3021f6aa.jpg 6
/work/projects/bigdata_sets/OpenImages_V4/train/61446dee2ee9eb27.jpg 7
/work/projects/bigdata_sets/OpenImages_V4/train/dba2da75d899c3e7.jpg 8
/work/projects/bigdata_sets/OpenImages_V4/train/7ea06f092abc005e.jpg 9
/work/projects/bigdata_sets/OpenImages_V4/train/2db694eba4d4bb04.jpg 10
```

The ULHPC team has designed a generic launcher for single node GNU parallel: see [`../basics/scripts/launcher.parallel.sh`](../basics/scripts/launcher.parallel.sh).
