-*- mode: markdown; mode: visual-line; fill-column: 80 -*-

Copyright (c) 2013-2017 UL HPC Team  <hpc-sysadmins@uni.lu>

        Time-stamp: <Tue 2017-06-20 11:56 svarrette>

------------------------------------------------------
# Running Big Data Application using Apache Spark on UL HPC platform

[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/advanced/Spark/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/advanced/Spark/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)


The objective of this tutorial is to compile and run on [Apache Spark](http://spark.apache.org/)  on top of the [UL HPC](http://hpc.uni.lu) platform.


**Advanced users only**: rely on `screen` (see  [tutorial](http://support.suso.com/supki/Screen_tutorial) or the [UL HPC tutorial](https://hpc.uni.lu/users/docs/ScreenSessions.html) on the  frontend prior to running any `oarsub` or `srun/sbatch` command to be more resilient to disconnection.

The latest version of this tutorial is available on
[Github](https://github.com/ULHPC/tutorials/tree/devel/advanced/Spark)

## Objectives

[Apache Spark](http://spark.apache.org/docs/latest/) is a large-scale data processing engine that performs in-memory computing. Spark offers bindings in Java, Scala, Python and R for building parallel applications.
high-level APIs in Java, Scala, Python and R, and an optimized engine that supports general execution graphs. It also supports a rich set of higher-level tools including Spark SQL for SQL and structured data processing, MLlib for machine learning, GraphX for graph processing, and Spark Streaming.

In this tutorial, we are going to build Spark using Easybuild and perform some basic examples.

## Building Spark

### Pre-requisite: Installing Easybuild

See also [PS3](../Easybuild).

First we are going to install Easybuild following [the official instructions](http://easybuild.readthedocs.io/en/latest/Installation.html).

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

### Search for an Easybuild Recipe for Spark

```bash
$> eb -S Spark
# Try to install the most recent version
$> eb Spark-2.0.2.eb -Dr    # Dry-run
$> eb Spark-2.0.2.eb -r
```

This is going to fail because of the Java dependency which is unable to build.
So we are going to create a custom easyconfig file


```bash
$> mkdir -p ~/tutorials/Spark
$> cd ~/tutorials/Spark
# Check the source easyconfig file
$> eb -S Spark
$> cp <path/to>/easyconfigs/s/Spark/Spark-2.0.2.eb  Spark-2.0.2.custom.eb
```

Modify it to ensure a successful build.

```diff
--- <path/to>/easyconfigs/s/Spark/Spark-2.0.2.eb 2017-06-12 22:16:14.353929000 +0200
+++ Spark-2.0.2.custom.eb 2017-06-12 22:39:59.155061000 +0200
@@ -15,7 +15,7 @@
     'http://www.us.apache.org/dist/%(namelower)s/%(namelower)s-%(version)s/',
 ]

-dependencies = [('Java', '1.7.0_80')]
+dependencies = [('Java', '1.8.0_121')]

 sanity_check_paths = {
     'files': ['bin/spark-shell'],
```

Build it and load the module

```bash
$> eb ./Spark-2.0.2.custom.eb
$> module spider Spark
$> module load devel/Spark
```

## Interactive usage

PySpark is the Spark Python API and exposes Spark Contexts to the Python programming environment. Use `--exclusive` to allocate an exclusive node, load the spark module, then run the Python Spark shell:

```bash
$> si --exclusive --ntasks=1 --cpus-per-task=28
$> module use $LOCAL_MODULE
$> module load devel/Spark
$> pyspark
```

After some initialization output, you will see the following:

```bash
$> pyspark
Python 2.7.5 (default, Nov  6 2016, 00:28:07)
[GCC 4.8.5 20150623 (Red Hat 4.8.5-11)] on linux2
Type "help", "copyright", "credits" or "license" for more information.
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel).
17/06/12 23:59:35 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 2.0.2
      /_/

Using Python version 2.7.5 (default, Nov  6 2016 00:28:07)
SparkSession available as 'spark'.
>>>
```

See [tutorial](https://www.dezyre.com/apache-spark-tutorial/pyspark-tutorial) for playing with pyspark.

## Scala Spark Shell

Spark includes a modified version of the Scala shell that can be used interactively. Instead of running `pyspark` above, run the `spark-shell` command:

```bash
$> spark-shell
```


After some initialization output, a scala shell prompt with the Spark context will be available:

```bash
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel).
17/06/13 00:06:43 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
17/06/13 00:06:43 WARN SparkContext: Use an existing SparkContext, some configuration may not take effect.
Spark context Web UI available at http://172.17.7.30:4040
Spark context available as 'sc' (master = local[*], app id = local-1497305203669).
Spark session available as 'spark'.
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /___/ .__/\_,_/_/ /_/\_\   version 2.0.2
      /_/

Using Scala version 2.11.8 (Java HotSpot(TM) 64-Bit Server VM, Java 1.8.0_121)
Type in expressions to have them evaluated.
Type :help for more information.

scala>
```

## R Spark Shell

The Spark R API is still experimental. Only a subset of the R API is available -- See the [SparkR Documentation](https://spark.apache.org/docs/latest/sparkr.html).

Load one of the R modules and then run the SparkR shell:

```bash
$> module load lang/R
$> sparkR

R version 3.4.0 (2017-04-21) -- "You Stupid Darkness"
Copyright (C) 2017 The R Foundation for Statistical Computing
Platform: x86_64-pc-linux-gnu (64-bit)

R is free software and comes with ABSOLUTELY NO WARRANTY.
You are welcome to redistribute it under certain conditions.
Type 'license()' or 'licence()' for distribution details.

  Natural language support but running in an English locale

R is a collaborative project with many contributors.
Type 'contributors()' for more information and
'citation()' on how to cite R or R packages in publications.

Type 'demo()' for some demos, 'help()' for on-line help, or
'help.start()' for an HTML browser interface to help.
Type 'q()' to quit R.

Launching java with spark-submit command /home/users/svarrette/.local/easybuild/software/devel/Spark/2.0.2/bin/spark-submit   "sparkr-shell" /tmp/Rtmphb0s8J/backend_port180ad365cc6b5
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel).
17/06/13 00:08:00 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable

 Welcome to
    ____              __
   / __/__  ___ _____/ /__
  _\ \/ _ \/ _ `/ __/  '_/
 /___/ .__/\_,_/_/ /_/\_\   version  2.0.2
    /_/


 SparkSession available as 'spark'.
>
```

## Running Spark standalone cluster

* [Reference Documentation](https://spark.apache.org/docs/latest/cluster-overview.html)


Spark applications run as independent sets of processes on a cluster, coordinated by the SparkContext object in your main program (called the driver program).

Specifically, to run on a cluster, the SparkContext can connect to several types of cluster managers (either Spark’s own standalone cluster manager, Mesos or YARN), which allocate resources across applications. Once connected, Spark acquires executors on nodes in the cluster, which are processes that run computations and store data for your application. Next, it sends your application code (defined by JAR or Python files passed to SparkContext) to the executors. Finally, SparkContext sends tasks to the executors to run.

![](https://spark.apache.org/docs/latest/img/cluster-overview.png)

There are several useful things to note about this architecture:

1. Each application gets its own executor processes, which stay up for the duration of the whole application and run tasks in multiple threads. This has the benefit of isolating applications from each other, on both the scheduling side (each driver schedules its own tasks) and executor side (tasks from different applications run in different JVMs). However, it also means that data cannot be shared across different Spark applications (instances of SparkContext) without writing it to an external storage system.
2. Spark is agnostic to the underlying cluster manager. As long as it can acquire executor processes, and these communicate with each other, it is relatively easy to run it even on a cluster manager that also supports other applications (e.g. Mesos/YARN).
3. The driver program must listen for and accept incoming connections from its executors throughout its lifetime (e.g., see spark.driver.port in the network config section). As such, the driver program must be network addressable from the worker nodes.
4. Because the driver schedules tasks on the cluster, it should be run close to the worker nodes, preferably on the same local area network. If you'd like to send requests to the cluster remotely, it's better to open an RPC to the driver and have it submit operations from nearby than to run a driver far away from the worker nodes.

**Cluster Manager**

Spark currently supports three cluster managers:

* [Standalone](https://spark.apache.org/docs/latest/spark-standalone.html) – a simple cluster manager included with Spark that makes it easy to set up a cluster.
* [Apache Mesos](https://spark.apache.org/docs/latest/running-on-mesos.html) – a general cluster manager that can also run Hadoop MapReduce and service applications.
* [Hadoop YARN](https://spark.apache.org/docs/latest/running-on-mesos.html) – the resource manager in Hadoop 2.

In this session, we will deploy a **standalone cluster**.

We will prepare a launcher script that will:

1. create a master and the workers
2. submit a spark application to the cluster using the `spark-submit` script
3. Let the application run and collect the result
4. stop the cluster at the end.

To facilitate these steps, Spark comes with a couple of scripts you can use to launch or stop your cluster, based on Hadoop's deploy scripts, and available in `$EBROOTSPARK/sbin`:

| Script                 | Description                                                                             |
|------------------------|------------------------------------------------------------------------------|
| `sbin/start-master.sh` | Starts a master instance on the machine the script is executed on.           |
| `sbin/start-slaves.sh` | Starts a slave instance on each machine specified in the conf/slaves file.   |
| `sbin/start-slave.sh`  | Starts a slave instance on the machine the script is executed on.            |
| `sbin/start-all.sh`    | Starts both a master and a number of slaves as described above.              |
| `sbin/stop-master.sh`  | Stops the master that was started via the bin/start-master.sh script.        |
| `sbin/stop-slaves.sh`  | Stops all slave instances on the machines specified in the conf/slaves file. |
| `sbin/stop-all.sh`     | Stops both the master and the slaves as described above.                     |

We are now going to prepare a launcher scripts to permit passive runs (typically in the `{default | batch}` queue).
We will place them in a separate directory as it will host the outcomes of the executions on the UL HPC platform .
Copy and adapt the [default SLURM launcher](https://github.com/ULHPC/launcher-scripts/blob/devel/slurm/launcher.default.sh) you should have a copy in `~/git/ULHPC/launcher-scripts/slurm/launcher.default.sh`


```bash
$> cd ~/tutorials/Spark/
$> cp ~/git/ULHPC/launcher-scripts/slurm/launcher.default.sh launcher-spark-pi.sh
```

The launcher will be organized as follows.


We will first exclusively allocate 2 nodes

```bash
#!/bin/bash -l
# Time-stamp: <Sun 2017-06-11 22:13 svarrette>
##################################################################
#SBATCH -N 1
# Exclusive mode is recommended for all spark jobs
#SBATCH --exclusive
#SBATCH --ntasks-per-node 1
### -c, --cpus-per-task=<ncpus>
###     (multithreading) Request that ncpus be allocated per process
#SBATCH -c 28
#SBATCH --time=0-01:00:00   # 1 hour
#
#          Set the name of the job
#SBATCH -J SparkMaster
#          Passive jobs specifications
#SBATCH --partition=batch
#SBATCH --qos qos-batch
```

Then we will load the custom module

```bash
# Use the RESIF build modules
if [ -f  /etc/profile ]; then
   .  /etc/profile
fi

# Load the {intel | foss} toolchain and whatever module(s) you need
module purge
module use $HOME/.local/easybuild/modules/all
module load devel/Spark

export SPARK_HOME=$EBROOTSPARK
```

Then start the Spark master and worker daemons using the Spark scripts

```bash
# sbin/start-master.sh - Starts a master instance on the machine the script is executed on.
$SPARK_HOME/sbin/start-all.sh

export MASTER=spark://$HOSTNAME:7077

echo
echo "========= Spark Master ========"
echo $MASTER
echo "==============================="
echo
```

Now we can submit an example python Pi estimation script to the Spark cluster with 100 partitions

_Note_: partitions in this context refers of course to Spark's Resilient Distributed Dataset (RDD) and how the dataset is distributed across the nodes in the Spark cluster.

```bash
spark-submit --master $MASTER  $SPARK_HOME/examples/src/main/python/pi.py 100
```

Finally, at the end, clean your environment and

```bash
# sbin/stop-master.sh - Stops the master that was started via the bin/start-master.sh script.
$SPARK_HOME/sbin/stop-all.sh
```

When the job completes, you can find the Pi estimation result in the slurm output file:

```
$> grep Pi slurm-2853.out
Pi is roughly 3.147861
```
