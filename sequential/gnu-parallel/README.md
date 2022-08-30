[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/sequential/basics/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/sequential/gnu-parallel/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/sequential/gnu-parallel/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Distributing embarrassingly parallel tasks GNU Parallel

     Copyright (c) 2021 UL HPC Team <hpc-team@uni.lu>

![](https://www.gnu.org/software/parallel/logo-gray+black300.png)

[GNU Parallel](http://www.gnu.org/software/parallel/) is a tool for executing tasks in parallel, typically on a single machine. **When coupled with the Slurm command `srun`, parallel becomes a powerful way of distributing a set of tasks amongst a number of workers**. This is particularly useful when the number of tasks is significantly larger than the number of available workers (i.e. `$SLURM_NTASKS`), and each tasks is independent of the others.

This tutorial is part of the practical session "[HPC Management of Sequential and Embarrassingly Parallel Jobs](https://ulhpc-tutorials.readthedocs.io/en/latest/sequential/basics/)". See the [associated slides](https://github.com/ULHPC/tutorials/raw/devel/sequential/basics/slides.pdf).


## Installation

The `parallel` command is available at the system level across the ULHPC clusters, yet under a relatively old version:

``` bash
(access)$ which parallel
/usr/bin/parallel
(access)$ parallel --version
GNU parallel 20190922
[...]
```

You may want to build the up-to-date version. The process is quite straight-forward and we will illustrate this process using the  [GNU Stow](https://www.gnu.org/software/stow/) utility which is quite useful for keeping track of system-wide and per-user installations of software built from source, as `parallel` in this case. [GNU Stow manual](https://www.gnu.org/software/stow/manual/stow.html#Introduction) - [tutorial](https://linuxconfig.org/how-to-use-gnu-stow-to-manage-programs-installed-from-source-and-dotfiles)

Prepare the installation directories within your HOME, together with the _stowdir_:

``` bash
### Access to ULHPC cluster if not yet done - here iris
(laptop)$> ssh aion-cluster
(access)$> cd      # go to your HOME
(access)$> mkdir -p bin include lib share/{doc,man} src
# create stowdir
(access)$> mkdir stow
```

Get the latest stable sources of [GNU Parallel](https://www.gnu.org/software/parallel/) under `src` and compile them within an interactive job:

``` bash
(access)$> cd ~/src
# Download the latest sources
(access)$> wget http://ftp.gnu.org/gnu/parallel/parallel-latest.tar.bz2
# Not strictly necessary, but help to keep track of time of action
(access)$> mv parallel-latest.tar.bz2  parallel-latest-$(date +%F).tar.bz2
# Uncompress
(access)$> tar xf parallel-latest-$(date +%F).tar.bz2
(access)$> cd parallel-20211022/
### Have an interactive job for the compilation process
# ... either directly
(access)$> si
# ... or using the HPC School reservation 'hpcschool' if needed  - use 'sinfo -T' to check if active and its name
# (access)$> si --reservation=hpcschool
```

[GNU Parallel](https://www.gnu.org/software/parallel/) is one of the many software that can be build very easily through the [Autotools](https://www.gnu.org/software/automake/manual/html_node/Autotools-Introduction.html) build system _i.e_ `./configure; make; make install`.
However, this process wants to install _by default_ the built software under `/usr/local` where you have **NO** rights to write files. So if you don't pay attention, the installation step will fail.
To circumvent the problem:

* we will install `parallel` in `$HOME` (`--prefix` option)
* more specifically, we will install the built software withing the stow directory, in a specific sub-directory that allow to specify the precise version generated.

Proceed as follows:

``` bash
(node)$> ./configure --prefix=$HOME/stow/parallel-20211022
(node)$> make
(node)$> make install
```

That's all folk.
You can now use `stow` to enable this build:

``` bash
(node)$> cd ~/stow
(node)$> stow parallel-20211022
# Check the result:
(node)$> ll ~/bin/parallel
lrwxrwxrwx. 1 svarrette clusterusers 38 Dec 12 14:50 /home/users/svarrette/bin/parallel -> ../stow/parallel-20211022/bin/parallel
# As ~/bin is part of your PATH at one of the first position, now the command
# 'parallel' is resolved as the newly built version
(node)$> which parallel
~/bin/parallel
(node)$>  parallel --version
GNU parallel 20211022   # In the unlikely case where you don't get the updated version, restart the bash session
[...]
```

At any moment of time, you can disable this build as follows:

``` bash
(node)$> cd ~/stow
(node)$> stow -D parallel-20211022
# Now the command 'parallel' is resolved to the system one
# IGNORE the 'BUG in find_stowed_path?' message due to https://github.com/aspiers/stow/issues/65
(node)$> ll ~/bin/parallel
ls: cannot access /home/users/svarrette/bin/parallel: No such file or directory
(node)$> which parallel
/usr/bin/parallel
(node)$> parallel --version     # you may need to source ~/.bashrc
GNU parallel 20190922
```

You can quit your interactive job (CTRL-D)


## Discovering the `parallel` command

The [GNU Parallel](http://www.gnu.org/software/parallel/) syntax can be a little distributing, but basically it supports two modes:

* Reading command arguments on the command line:

        parallel	[-j N] [OPTIONS]	COMMAND	{} ::: TASKLIST

* Reading command arguments from an input file:

        parallel	–a	TASKLIST.LST	[-j N] [OPTIONS]	COMMAND	{}
        parallel	[-j N] [OPTIONS]	COMMAND	{} :::: TASKLIST.LST

If your COMMAND embed a pipe stage, you have to escape the pipe symbol as follows `\|`.
Let's make some tests. The `-j <N>` option permits to define the jobs per machine - in particular you may want to use `-j 1` to enable a sequential resolution of the parallel command

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

The ULHPC team has designed a generic launcher for GNU parallel: see [`../basics/scripts/launcher.parallel.sh`](../basics/scripts/launcher.parallel.sh).

Its usage is explicited in the [HPC Management of Sequential and Embarrassingly Parallel Jobs](../basics/) tutorials.
