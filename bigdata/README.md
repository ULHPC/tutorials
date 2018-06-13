[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/bigdata/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/bigdata/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/bigdata/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)


# Big Data Applications (batch, stream, hybrid)

     Copyright (c) 2013-2018 UL HPC Team  <hpc-sysadmins@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/bigdata/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/bigdata/slides.pdf)

The objective of this tutorial is to demonstrate how to build and run on top of the [UL HPC](http://hpc.uni.lu) platform a couple of reference analytics engine for large-scale Big Data processing, _i.e._ [Hadoop](http://hadoop.apache.org/) or  [Apache Spark](http://spark.apache.org/)


--------------------
## Pre-requisites ##

Ensure you are able to [connect to the UL HPC clusters](https://hpc.uni.lu/users/docs/access.html)
**For all tests and compilation, you MUST work on a computing node**

You'll need to prepare the data sources required by this tutorial once connected

``` bash
### ONLY if not yet done: setup the tutorials repo
# See http://ulhpc-tutorials.rtfd.io/en/latest/setup/install/
$> mkdir -p ~/git/github.com/ULHPC
$> cd ~/git/github.com/ULHPC
$> git clone https://github.com/ULHPC/tutorials.git
$> cd tutorials
$> make setup          # Initiate git submodules etc...
```

Now you can prepare a dedicated directory to work on this tutorial:

```bash
$> cd ~/git/github.com/ULHPC/tutorials/bigdata
```

**Advanced users only**: rely on `screen` (see  [tutorial](http://support.suso.com/supki/Screen_tutorial) or the [UL HPC tutorial](https://hpc.uni.lu/users/docs/ScreenSessions.html) on the  frontend prior to running any `oarsub` or `srun/sbatch` command to be more resilient to disconnection.

Finally, be aware that the latest version of this tutorial is available on
[Github](https://github.com/ULHPC/tutorials/tree/devel/bigdata/) and on

<http://ulhpc-tutorials.readthedocs.io/en/latest/bigdata/>

One of the first objective is to install the  [Hadoop MapReduce by Cloudera](https://www.cloudera.com/downloads/cdh/5-12-0.html) using [EasyBuild](http://easybuild.readthedocs.io/).

**`/!\ IMPORTANT`**: For this reason, it is advised to have first followed the [Easybuild tutorial in `tools/easybuild.md`](http://ulhpc-tutorials.readthedocs.io/en/latest/tools/easybuild/). **In all cases,** you need to have easybuild locally installed -- see [installation guidelines](http://ulhpc-tutorials.readthedocs.io/en/latest/tools/easybuild/#part-2-easybuild).


In the next part, we are going to install a few mandatory software required to install and use [Hadoop](http://hadoop.apache.org/) or  [Apache Spark](http://spark.apache.org/).

----------------------------------
## 1. Preliminary installations ##

As this involves Java (something more probably HPC users don't like), and that Java needs to be treated specifically within Easybuild do to the licences involved, we will now cover it.

For these builds, reserve an interactive job on one full node (for 3 hours)

```bash
############### iris cluster (slurm) ###############
(access-iris)$> srun -p interactive -N 1 -n 1 -c 28 -t 3:00:00 --pty bash

############### gaia/chaos clusters (OAR) ###############
(access-{gaia|chaos})$> oarsub -I -l nodes=1,walltime=3
```

### Step 1.a. Java 7u80 and 8u152

We'll need several version of the [JDK](http://www.oracle.com/technetwork/java/javase/overview/index.html) (in Linux x64 source mode i.e. `jdk-<version>-linux-x64.tar.gz`), more specifically 1.7.0_80 (aka `7u80` in Oracle's naming convention) and 1.8.0_152 (aka `8u152`).

Let's first try the classical approach we experimented before:

``` bash
$> module avail java

---------------- /opt/apps/resif/data/stable/default/modules/all --------------
   lang/Java/1.8.0_121
```

Either an older (1.7.0_80) or more recent Java (1.8.0_152) is required, so let's search for Java and install it:

``` bash
$> mu      # See Easybuild tutorial
$> eb -S Java-1.7
[...]
* $CFGS1/j/Java/Java-1.7.0_80.eb
[...]
$> eb Java-1.7.0_80.eb -Dr
== temporary log file in case of crash /tmp/eb-MJtqvZ/easybuild-L1NfjN.log
Dry run: printing build status of easyconfigs and dependencies
CFGS=/home/users/svarrette/.local/easybuild/software/tools/EasyBuild/3.6.1/lib/python2.7/site-packages/easybuild_easyconfigs-3.6.1-py2.7.egg/easybuild/easyconfigs/j/Java
 * [ ] $CFGS/Java-1.7.0_80.eb (module: lang/Java/1.7.0_80)
== Temporary log file(s) /tmp/eb-MJtqvZ/easybuild-L1NfjN.log* have been removed.
== Temporary directory /tmp/eb-MJtqvZ has been removed.

$> time eb Java-1.7.0_80.eb -r
== temporary log file in case of crash /tmp/eb-wkXTsu/easybuild-UqzX_P.log
== resolving dependencies ...
== processing EasyBuild easyconfig /home/users/svarrette/.local/easybuild/software/tools/EasyBuild/3.6.1/lib/python2.7/site-packages/easybuild_easyconfigs-3.6.1-py2.7.egg/easybuild/easyconfigs/j/Java/Java-1.7.0_80.eb
== building and installing lang/Java/1.7.0_80...
== fetching files...
== FAILED: Installation ended unsuccessfully (build directory: /home/users/svarrette/.local/easybuild/build/Java/1.7.0_80/dummy-dummy): build failed (first 300 chars): Couldn't find file jdk-7u80-linux-x64.tar.gz anywhere, and downloading it didn't work either...
[...]
```

As the error indicates, you first need to download the archive.
Hopefully, the main `Makefile` helps to collect the archives:

``` bash
$> pwd
/home/users/<login>/git/hub.com/ULHPC/tutorials/bigdata
$> make fetch
./scripts/bootstrap.sh --java7 --output-dir src
==> Downloading Java 7 archive 'jdk-7u80-linux-x64.tar.gz'

100%[======================================================================================>] 153,530,841 4.44MB/s   in 54s

./scripts/bootstrap.sh --java8 --output-dir src
==> Downloading Java 8 archive 'jdk-8u152-linux-x64.tar.gz'
[...]

100%[======================================================================================>] 189,784,266 24.5MB/s   in 25s

2018-06-12 21:40:32 (7.22 MB/s) - ‘jdk-8u152-linux-x64.tar.gz’ saved [189784266/189784266]
```

The sources are saved in `src/`

```bash
$> cd src
$> eb Java-1.7.0_80.eb -Dr  # Dry-run
# Real run -- set robot path to search in the local directory (do not forget the ':').
# Prefix by time to get the time required to execute the command.
$> time eb Java-1.7.0_80.eb --robot-paths=$PWD: -r

# Repeat for Java 8:
$> eb -S Java-1.8
[...]
* $CFGS2/j/Java/Java-1.8.0_152.eb
[...]
$> eb Java-1.8.0_152.eb -Dr # Dry-run
$> time eb Java-1.8.0_152.eb --robot-paths=$PWD: -r
```

Check the result:

```bash
$> module av java
$> module show lang/Java/1.7.0_80
```

### Step 1.b. Maven 3.5.2

We will also need an updated version of [Maven](https://maven.apache.org/) (3.5.2).

Let's first try with the default reciPy/easyconfig:

```bash
$> eb -S Maven
[...]
 * $CFGS1/Maven-3.2.3.eb
 * $CFGS1/Maven-3.3.3.eb
 * $CFGS1/Maven-3.3.9.eb
 * $CFGS1/Maven-3.5.0.eb
 * $CFGS1/Maven-3.5.2.eb

# Let's try to install the most recent one:
$> eb Maven-3.5.2.eb -Dr
[...]
 * [ ] $CFGS/Maven-3.5.2.eb (module: devel/Maven/3.5.0)
$> time eb Maven-3.5.2.eb -r
```

Check the result:

```bash
$> module av Maven
module av Maven
----------- /home/users/<login>/.local/easybuild/modules/all --------
   devel/Maven/3.5.2
```

A few other elements need to be installed.

### Step 1.c CMake, snappy, protobuf

Let's repeat the process globally for:

* [Cmake](https://cmake.org/) **3.9.1**  (the version is important), a very popular an open-source, cross-platform family of tools designed to build, test and package software,  * [snappy](https://github.com/google/snappy)
**version 1.1.6** (the version is important), the fast compressor/decompressor from Google, and
* [protobuf](https://github.com/google/protobuf), Google's language-neutral, platform-neutral, extensible mechanism for serializing structured data (**version 2.5.0**) we'll need later:

```bash
$> eb CMake-3.9.1.eb -Dr
$> time eb CMake-3.9.1.eb -r
[...]
real    6m51.780s
user    5m12.837s
sys     1m10.029s

$> eb snappy-1.1.6.eb -Dr
$> time snappy-1.1.6.eb -r
[...]
real    0m7.315s
user    0m3.768s
sys     0m1.918s


$> eb protobuf-2.5.0.eb -Dr    # Dry-run
$> time eb protobuf-2.5.0.eb -r
[...]
real    1m51.002s
user    1m35.820s
sys     0m11.222s
```


----------------------------
## 2. Hadoop Installation ##

We're going to install the most recent  [Hadoop by Cloudera](https://www.cloudera.com/downloads/cdh/5-12-0.html) _i.e._ `Hadoop-2.6.0-cdh5.12.0-native.eb`.

```bash
$> eb -S Hadoop | grep cdh
 * $CFGS2/h/Hadoop/Hadoop-2.5.0-cdh5.3.1-native.eb
 * $CFGS2/h/Hadoop/Hadoop-2.6.0-cdh5.12.0-native.eb
 * $CFGS2/h/Hadoop/Hadoop-2.6.0-cdh5.4.5-native.eb
 * $CFGS2/h/Hadoop/Hadoop-2.6.0-cdh5.7.0-native.eb
 * $CFGS2/h/Hadoop/Hadoop-2.6.0-cdh5.8.0-native.eb
```

To have it _successfully_ built, we'll just need to adapt the corresponding recipY to use the latest Maven we just installed.

```bash
$> eb -S Hadoop-2.6.0-cdh5.12.0-native.eb
CFGS1=$HOME/.local/easybuild/software/tools/EasyBuild/3.6.1/lib/python2.7/site-packages/easybuild_easyconfigs-3.6.1-py2.7.egg/easybuild/easyconfigs/h/Hadoop
 * $CFGS1/Hadoop-2.6.0-cdh5.12.0-native.eb

# Copy the recipy -- you need thus to define the CDGS1 variable
$> CFGS1=$HOME/.local/easybuild/software/tools/EasyBuild/3.6.1/lib/python2.7/site-packages/easybuild_easyconfigs-3.6.1-py2.7.egg/easybuild/easyconfigs/h/Hadoop
$> echo $CFGS1
$> cd /tmp     # Work in a temporary directory
$> cp $CFGS1/Hadoop-2.6.0-cdh5.12.0-native.eb .
```
Now adapt the recipY to use the latest Maven we just installed.

```diff
--- $HOME/.local/easybuild/software/tools/EasyBuild/3.6.1/lib/python2.7/site-packages/easybuild_easyconfigs-3.6.1-py2.7.egg/easybuild/easyconfigs/h/Hadoop/Hadoop-2.6.0-cdh5.12.0-native.eb     2018-06-07 23:30:47.111972000 +0200
+++ /tmp/Hadoop-2.6.0-cdh5.12.0-native.eb    2018-06-12 22:44:26.425625000 +0200
@@ -14,7 +14,7 @@
 patches = ['Hadoop-TeraSort-on-local-filesystem.patch']

 builddependencies = [
-    ('Maven', '3.5.0'),
+    ('Maven', '3.5.2'),
     ('protobuf', '2.5.0'),  # *must* be this version
     ('CMake', '3.9.1'),
     ('snappy', '1.1.6'),
```

_Note_: the resulting Easyconfigs is provided to you in `src/Hadoop-2.6.0-cdh5.12.0-native.eb`:

```bash
$> module load devel/Maven devel/protobuf/2.5.0  devel/CMake/3.9.1 lib/snappy/1.1.6
$> module list

Currently Loaded Modules:
1) tools/EasyBuild/3.6.1   3) devel/Maven/3.5.2      5) devel/CMake/3.9.1
2) lang/Java/1.7.0_80      4) devel/protobuf/2.5.0   6) lib/snappy/1.1.6
```

Let's install it:

```
$> eb ./Hadoop-2.6.0-cdh5.12.0-native.eb -Dr   # Dry run
$> time eb ./Hadoop-2.6.0-cdh5.12.0-native.eb -r
[...]
real    52m58.484s
user    5m28.819s
sys     2m8.673s
```

**`/!\ IMPORTANT`: As you can see, the build is quite long -- it takes ~53 minutes**
You can monitor the execution in parallel using `htop`


-----------------------
## 3. Running Hadoop ##

```
$> module av Hadoop
$> module load tools/Hadoop
```

When doing that, the Hadoop distribution is installed in `$EBROOTHADOOP` (this is set by Easybuild for any loaded software.)

The below instructions are based on the [official tutorial](https://hadoop.apache.org/docs/r2.6.0/hadoop-project-dist/hadoop-common/SingleCluster.html).

### 3.a Hadoop in Single mode

By default, Hadoop is configured to run in a non-distributed mode, as a single Java process. This is useful for debugging.

Let's test it

```bash
$> cd runs/hadoop/single
# Prepare input data
$> mkdir input
$> cp ${EBROOTHADOOP}/etc/hadoop/*.xml input
# Map-reduce grep <pattern> -- result is produced in output/
$> hadoop jar ${EBROOTHADOOP}/share/hadoop/mapreduce/hadoop-mapreduce-examples-2.6.0-cdh5.12.0.jar grep input output 'dfs[a-z.]+'
[...]
        File System Counters
                FILE: Number of bytes read=70426
                FILE: Number of bytes written=1202298
                FILE: Number of read operations=0
                FILE: Number of large read operations=0
                FILE: Number of write operations=0
        Map-Reduce Framework
                Map input records=1
                Map output records=1
                Map output bytes=17
                Map output materialized bytes=25
                Input split bytes=186
                Combine input records=0
                Combine output records=0
                Reduce input groups=1
                Reduce shuffle bytes=25
                Reduce input records=1
                Reduce output records=1
                Spilled Records=2
                Shuffled Maps =1
                Failed Shuffles=0
                Merged Map outputs=1
                GC time elapsed (ms)=8
                Total committed heap usage (bytes)=1046478848
        Shuffle Errors
                BAD_ID=0
                CONNECTION=0
                IO_ERROR=0
                WRONG_LENGTH=0
                WRONG_MAP=0
                WRONG_REDUCE=0
        File Input Format Counters
                Bytes Read=123
        File Output Format Counters
                Bytes Written=23
# Check the results
$> cat output/*
1       dfsadmin
```

### 3.b Pseudo-Distributed Operation

Hadoop can also be run on a single-node in a pseudo-distributed mode where each Hadoop daemon runs in a separate Java process.
Follow the [official tutorial](https://hadoop.apache.org/docs/r2.6.0/hadoop-project-dist/hadoop-common/SingleCluster.html#Pseudo-Distributed_Operation) to ensure you are running in **Single Node Cluster**

Once this is done, follow the [official Wordcount instructions](https://hadoop.apache.org/docs/r2.6.0/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html#Example:_WordCount_v1.0)

### 3.b Full cluster setup

Follow the official instructions of the [Cluster Setup](https://hadoop.apache.org/docs/r2.6.0/hadoop-project-dist/hadoop-common/ClusterSetup.html).

Once this is done, Repeat the execution of the [official Wordcount example](https://hadoop.apache.org/docs/r2.6.0/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html#Example:_WordCount_v1.0).

--------------------------------------------------
## 4. Interactive Big Data Analytics with Spark ##

The objective of this section is to compile and run on [Apache Spark](http://spark.apache.org/)  on top of the [UL HPC](http://hpc.uni.lu) platform.

[Apache Spark](http://spark.apache.org/docs/latest/) is a large-scale data processing engine that performs in-memory computing. Spark offers bindings in Java, Scala, Python and R for building parallel applications.
high-level APIs in Java, Scala, Python and R, and an optimized engine that supports general execution graphs. It also supports a rich set of higher-level tools including Spark SQL for SQL and structured data processing, MLlib for machine learning, GraphX for graph processing, and Spark Streaming.

As for Hadoop, we are first going to build Spark using Easybuild before performing some basic examples. More precisely, in this part, we will review the basic usage of Spark in two cases:

1. a single conffiguration where the classical interactive wrappers (`pyspark`, `scala` and `R` wrappers) will be reviewed.
2. a [Standalone](https://spark.apache.org/docs/latest/spark-standalone.html) cluster configuration - a simple cluster manager included with Spark that makes it easy to set up a cluster), where we will run the Pi estimation.

### 4.1 Building Spark

Spark 2.1.1 is available by default.

We are still going to use a more recent version:

```bash
$> eb -S Spark
[...]
 * $CFGS2/Spark-2.2.0-Hadoop-2.6-Java-1.8.0_152.eb

# Try to install one of the most recent version
$> eb Spark-2.2.0-Hadoop-2.6-Java-1.8.0_152.eb -Dr
$> time eb Spark-2.2.0-Hadoop-2.6-Java-1.8.0_152.eb -r
[...]
real    0m9.940s
user    0m5.167s
sys     0m2.475s
```

### 4.2 Interactive usage

Exit your reservation to reload one with the `--exclusive` flag to allocate an exclusive node.
Let's load the installed module:

```bash
$> srun -p interactive -c 28 --exclusive -t 2:00:00 --pty bash
$> mu
$> module av Spark

---------- /home/users/svarrette/.local/easybuild/modules/all ----------
   devel/Spark/2.2.0-Hadoop-2.6-Java-1.8.0_152 (D)

------------ /opt/apps/resif/data/stable/default/modules/all -----------
   devel/Spark/2.1.1

$> module load devel/Spark
```

#### 4.2.a. Pyspark

PySpark is the Spark Python API and exposes Spark Contexts to the Python programming environment.

```bash
pyspark
Python 2.7.5 (default, Aug  4 2017, 00:39:18)
[GCC 4.8.5 20150623 (Red Hat 4.8.5-16)] on linux2
Type "help", "copyright", "credits" or "license" for more information.
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
18/06/13 00:51:28 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
18/06/13 00:51:33 WARN ObjectStore: Version information not found in metastore. hive.metastore.schema.verification is not enabled so recording the schema version 1.2.0
18/06/13 00:51:33 WARN ObjectStore: Failed to get database default, returning NoSuchObjectException
18/06/13 00:51:34 WARN ObjectStore: Failed to get database global_temp, returning NoSuchObjectException
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 2.2.0
      /_/

Using Python version 2.7.5 (default, Aug  4 2017 00:39:18)
SparkSession available as 'spark'.
>>>
```

See [this tutorial](https://www.dezyre.com/apache-spark-tutorial/pyspark-tutorial) for playing with pyspark.

#### 4.2.b. Scala Spark Shell

Spark includes a modified version of the Scala shell that can be used interactively.
Instead of running `pyspark` above, run the `spark-shell` command:

```bash
$> spark-shell
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
[...]
Spark context Web UI available at http://172.17.6.1:4040
Spark context available as 'sc' (master = local[*], app id = local-1528844025861).
Spark session available as 'spark'.
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /___/ .__/\_,_/_/ /_/\_\   version 2.2.0
      /_/

Using Scala version 2.11.8 (Java HotSpot(TM) 64-Bit Server VM, Java 1.8.0_152)
Type in expressions to have them evaluated.
Type :help for more information.

scala>
```

#### 4.2.c.  R Spark Shell

The Spark R API is still experimental. Only a subset of the R API is available -- See the [SparkR Documentation](https://spark.apache.org/docs/latest/sparkr.html).
Since this tutorial does not cover R, we are not going to use it.


### 4.3 Running Spark in standalone cluster

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

In this session, we will deploy a **standalone cluster**, which consists of performing the following workflow (with the objective to prepare a launcher script):

1. create a master and the workers. Check the web interface of the master.
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

Exit (if needed) the previous session, and make a new reservation:

```bash
``bash
$> srun -p interactive -n 4 -c 7 --exclusive --pty bash
$> mu
$> module av Spark

---------- /home/users/svarrette/.local/easybuild/modules/all ----------
   devel/Spark/2.2.0-Hadoop-2.6-Java-1.8.0_152 (D)

------------ /opt/apps/resif/data/stable/default/modules/all -----------
   devel/Spark/2.1.1

$> module load devel/Spark
```

### Creation of a master

Let's first start a master process:

```bash
$> start-master.sh -h
Usage: ./sbin/start-master.sh [options]
18/06/13 01:16:34 INFO Master: Started daemon with process name: 37750@iris-001
18/06/13 01:16:34 INFO SignalUtils: Registered signal handler for TERM
18/06/13 01:16:34 INFO SignalUtils: Registered signal handler for HUP
18/06/13 01:16:34 INFO SignalUtils: Registered signal handler for INT

Options:
  -i HOST, --ip HOST     Hostname to listen on (deprecated, please use --host or -h)
  -h HOST, --host HOST   Hostname to listen on
  -p PORT, --port PORT   Port to listen on (default: 7077)
  --webui-port PORT      Port for web UI (default: 8080)
  --properties-file FILE Path to a custom Spark properties file.
                         Default is conf/spark-defaults.conf.

$> start-master.sh --host $(hostname)
```

Unlike what claim the help message, the `start-master.sh` script will launch a web interface on the port 8082 i.e. on `http://$(hostname):8082`

You can check it:

```bash
$> netstat -an 8082
```

We are going to access this web portal (on `http://<IP>:8082`) using a SOCKS 5 Proxy Approach.
That means that:

* You should initiate an SSH connetion with `-D 1080` option to open on the local port 1080:

```
(laptop)$> ssh -D 1080 -C iris-cluster
```
Now, install for example the [Foxy Proxy](https://getfoxyproxy.org/order/?src=FoxyProxyForFirefox)
extension for Firefox and configure it to use your SOCKS proxy:

* Right click on the fox icon
* Options
* **Add a new proxy** button
* Name: `ULHPC proxy`
* Informations > **Manual configuration**
  * Host IP: `127.0.0.1`
  * Port: `1080`
  * Check the **Proxy SOCKS** Option
* Click on **OK**
* Close
* Open a new tab
* Right click on the Fox
* Choose the **ULHPC proxy**

Now you should be able to access the Spark master website, by entering the URL `http://172.17.XX.YY:8082/` (adapt the IP).

When you have finished, don't forget to close your tunnel and disable FoxyProxy
on your browser.


### Creation of a worker

```bash
$> export SPARK_HOME=$EBROOTSPARK           # Required
$> export MASTER=spark://$(hostname):7077   # Helpful
$> echo $MASTER
```
Now we can start a worker:

```bash
$> start-slave.sh -h
Usage: ./sbin/start-slave.sh [options] <master>
18/06/13 01:57:54 INFO Worker: Started daemon with process name: 44910@iris-001
18/06/13 01:57:54 INFO SignalUtils: Registered signal handler for TERM
18/06/13 01:57:54 INFO SignalUtils: Registered signal handler for HUP
18/06/13 01:57:54 INFO SignalUtils: Registered signal handler for INT

Master must be a URL of the form spark://hostname:port

Options:
  -c CORES, --cores CORES  Number of cores to use
  -m MEM, --memory MEM     Amount of memory to use (e.g. 1000M, 2G)
  -d DIR, --work-dir DIR   Directory to run apps in (default: SPARK_HOME/work)
  -i HOST, --ip IP         Hostname to listen on (deprecated, please use --host or -h)
  -h HOST, --host HOST     Hostname to listen on
  -p PORT, --port PORT     Port to listen on (default: random)
  --webui-port PORT        Port for web UI (default: 8081)
  --properties-file FILE   Path to a custom Spark properties file.
                           Default is conf/spark-defaults.conf.

$> start-slave.sh -c ${SLURM_CPUS_PER_TASK} $MASTER
```

Check the result on the master website `http://<IP>:8082`.

Now we can submit an example python Pi estimation script to the Spark cluster with 100 partitions

_Note_: partitions in this context refers of course to Spark's Resilient Distributed Dataset (RDD) and how the dataset is distributed across the nodes in the Spark cluster.

```bash
$> spark-submit --master $MASTER  $SPARK_HOME/examples/src/main/python/pi.py 100
[...]
18/06/13 02:03:43 INFO DAGScheduler: Job 0 finished: reduce at /home/users/svarrette/.local/easybuild/software/devel/Spark/2.2.0-Hadoop-2.6-Java-1.8.0_152/examples/src/main/python/pi.py:43, took 3.738313 s
Pi is roughly 3.140860
```

Finally, at the end, clean your environment and

```bash
# sbin/stop-master.sh - Stops the master that was started via the bin/start-master.sh script.
$SPARK_HOME/sbin/stop-all.sh
```

Prepare a launcher.
See also [Spark launcher example](https://hpc.uni.lu/users/docs/slurm_launchers.html#apache-spark).
