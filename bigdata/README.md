[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/bigdata/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/bigdata/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/bigdata/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)


# Big Data Applications (batch, stream, hybrid)

     Copyright (c) 2013-2020 UL HPC Team  <hpc-team@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/bigdata/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/bigdata/slides.pdf)

The objective of this tutorial is to demonstrate how to build and run on top of the [UL HPC](http://hpc.uni.lu) platform a couple of reference analytics engine for large-scale Big Data processing, _i.e._ [Hadoop](http://hadoop.apache.org/) or  [Apache Spark](http://spark.apache.org/).


--------------------
## Pre-requisites ##

Ensure you are able to [connect to the UL HPC clusters](https://hpc.uni.lu/users/docs/access.html)
In particular, recall that the `module` command **is not** available on the access frontends.
**For all tests, builds and compilation, you MUST work on a computing node**

```bash
### Access to ULHPC cluster - here iris
(laptop)$> ssh iris-cluster
```

Now you'll need to pull the latest changes in your working copy of the [ULHPC/tutorials](https://github.com/ULHPC/tutorials) you should have cloned in `~/git/github.com/ULHPC/tutorials` (see ["preliminaries" tutorial](../preliminaries/))

``` bash
(access)$> cd ~/git/github.com/ULHPC/tutorials
(access)$> git pull
```

Now **configure a dedicated directory `~/tutorials/bigdata` for this session**

``` bash
# return to your home
(access)$> mkdir -p ~/tutorials/bigdata
(access)$> cd ~/tutorials/bigdata
# create a symbolic link to the reference material
(access)$> ln -s ~/git/github.com/ULHPC/tutorials/bigdata ref.d
# Prepare a couple of symbolic links that will be useful for the training
(access)$> ln -s ref.d/scripts .     # Don't forget trailing '.' means 'here'
(access)$> ln -s ref.d/settings .    # idem
(access)$> ln -s ref.d/src .         # idem
```

**Advanced users** (_eventually_ yet __strongly__ recommended), create a [GNU Screen](http://www.gnu.org/software/screen/) session you can recover later - see ["Getting Started" tutorial](../getting-started/) or [this `screen` tutorial](http://support.suso.com/supki/Screen_tutorial).

### Easybuild

One of the first objective is to install the latest version of [Hadoop](https://hadoop.apache.org/releases.html).
using [EasyBuild](http://easybuild.readthedocs.io/).
For this reason, you should first check the [Easybuild tutorial in `tools/easybuild.md`](http://ulhpc-tutorials.readthedocs.io/en/latest/tools/easybuild/) and install the latest version of Easybuild (4.3.2 at the time of writing).

Note that it should be sufficient to run the following command **once on a node**

``` bash
### Have an interactive job
# ... either directly
(access)$> si
# ... or using the HPC School reservation 'hpcschool' if needed  - use 'sinfo -T' to check if active and its name
# (access)$> srun --reservation=hpcschool --pty bash
(node)$> ~/git/github.com/ULHPC/tutorials/tools/easybuild/scripts/setup.sh -h  # Help - check EASYBUILD_*
(node)$> ~/git/github.com/ULHPC/tutorials/tools/easybuild/scripts/setup.sh -n  # Dry-run
(node)$> ~/git/github.com/ULHPC/tutorials/tools/easybuild/scripts/setup.sh     # install
```


### 2019b software set

As indicated in the keynote, the 2019b software set is available for general availability and for **testing** purposes until Jan 31, 2021. We will use it in this tutorial, yes as it is not enabled by default, you will have to **setup it each time you request an interactive job**.
Proceed as follows:

```bash
### Have an interactive job
(access)$> si
# Enable (new) 2019b software set - iris cluster
(node)$> unset MODULEPATH   # Safeguard to avoid mixing up with 2019a software
(node)$> module use /opt/apps/resif/iris/2019b/broadwell/modules/all
# Note: we won't use it here but later you may need to use the skylake/GPU-specialized builds,
#       which should be used **ONLY** on skylake/GPU nodes
# module use /opt/apps/resif/iris/2019b/skylake/modules/all
# module use /opt/apps/resif/iris/2019b/gpu/modules/all

### Now check that you have the latest EB 4.3.2 installed
#                 # assuming you hade defined:   export LOCAL_MODULES=${EASYBUILD_PREFIX}/modules/all
(node)$> mu       # shortcut for module use $LOCAL_MODULES; module load tools/EasyBuild
(node)$> eb --version
This is EasyBuild 4.3.1 (framework: 4.3.2, easyblocks: 4.3.2) on host iris-131.
(node)$> echo $MODULEPATH
/home/users/<login>/.local/easybuild/modules/all:/opt/apps/resif/iris/2019b/broadwell/modules/all
# If all OK: you should be able to access Spark module 2.4.3 for 2019b toolchain
(node)$> module avail Spark

----------- /opt/apps/resif/iris/2019b/broadwell/modules/all ----------------
   devel/Spark/2.4.3-intel-2019b-Hadoop-2.7-Java-1.8-Python-3.7.4

[...]
```

As this procedure will have to be repeated several time, you can make it done by sourcing `settings/2019b`

```bash
(access)$> si
(node)$> source settings/2019b
# Double-check
(node)$> eb --version
This is EasyBuild 4.3.2 (framework: 4.3.2, easyblocks: 4.3.2) on host iris-117
(node)$> echo $MODULEPATH
/home/users/<login>/.local/easybuild/modules/all:/opt/apps/resif/iris/2019b/broadwell/modules/all
```

In the next part, we are going to install a few mandatory software required to install and use [Hadoop](http://hadoop.apache.org/) or  [Apache Spark](http://spark.apache.org/).


### SOCKS 5 Proxy plugin (optional but VERY useful)

Many Big Data framework involves a web interface (at the level of the master and/or the workers) you probably want to access in a relative transparent way.

For that, a convenient way is to rely on a SOCKS proxy, which is basically an SSH tunnel in which specific applications forward their traffic down the tunnel to the server, and then on the server end, the proxy forwards the traffic out to the general Internet. Unlike a VPN, a SOCKS proxy has to be configured on an app by app basis on the client machine, but can be set up without any specialty client agents.

These steps were also described in the [Preliminaries](../preliminaries) tutorial.

__Setting Up the Tunnel__

To initiate such a SOCKS proxy using SSH (listening on `localhost:1080` for instance), you simply need to use the `-D 1080` command line option when connecting to the cluster:

```bash
(laptop)$> ssh -D 1080 -C iris-cluster
```

* `-D`: Tells SSH that we want a SOCKS tunnel on the specified port number (you can choose a number between 1025-65536)
* `-C`: Compresses the data before sending it

__Configuring Firefox to Use the Tunnel__: see [Preliminaries](../preliminaries) tutorial

We will see later on (in the section dedicated to Spark) how to effectively use this configuration.


-------------------------------
## Getting Started with Hadoop

### Installation

Quit your precedent job (`CTRL-D`) and let's reserve a new one with more cores to accelerate the builds:

```bash
(access)$> si -c 14     # In normal times: target all cores i.e. 28
(node)$> source settings/2019b
# (node)$> mu    # not necessary but kept for your information
(node)$> eb --version
This is EasyBuild 4.3.2 (framework: 4.3.2, easyblocks: 4.3.2) on host iris-117
# Search for a recent version of Hadoop
$> eb -S Hadoop-2
== found valid index for /home/users/svarrette/.local/easybuild/software/EasyBuild/4.3.1/easybuild/easyconfigs, so using it...
CFGS1=/opt/apps/resif/data/easyconfigs/ulhpc/default/easybuild/easyconfigs/s/Spark
CFGS2=/home/users/svarrette/.local/easybuild/software/EasyBuild/4.3.1/easybuild/easyconfigs
 * $CFGS1/Spark-2.3.0-intel-2018a-Hadoop-2.7-Java-1.8.0_162-Python-3.6.4.eb
 * $CFGS1/Spark-2.4.0-Hadoop-2.7-Java-1.8.eb
 * $CFGS1/Spark-2.4.3-intel-2019a-Hadoop-2.7-Java-1.8-Python-3.7.2.eb
 * $CFGS2/h/Hadoop/Hadoop-2.10.0-GCCcore-8.3.0-native.eb
 * $CFGS2/h/Hadoop/Hadoop-2.10.0_tirpc.patch
 * $CFGS2/h/Hadoop/Hadoop-2.4.0-seagate-722af1-native.eb
 * $CFGS2/h/Hadoop/Hadoop-2.5.0-cdh5.3.1-native.eb
 * $CFGS2/h/Hadoop/Hadoop-2.6.0-cdh5.12.0-native.eb
 * $CFGS2/h/Hadoop/Hadoop-2.6.0-cdh5.4.5-native.eb
 * $CFGS2/h/Hadoop/Hadoop-2.6.0-cdh5.7.0-native.eb
 * $CFGS2/h/Hadoop/Hadoop-2.6.0-cdh5.8.0-native.eb
 * $CFGS2/h/Hadoop/Hadoop-2.9.2-GCCcore-7.3.0-native.eb
 * $CFGS2/h/Hadoop/Hadoop-2.9.2_fix-zlib.patch
 * $CFGS2/s/Spark/Spark-2.2.0-Hadoop-2.6-Java-1.8.0_144.eb
 * $CFGS2/s/Spark/Spark-2.2.0-Hadoop-2.6-Java-1.8.0_152.eb
 * $CFGS2/s/Spark/Spark-2.2.0-intel-2017b-Hadoop-2.6-Java-1.8.0_152-Python-3.6.3.eb
 * $CFGS2/s/Spark/Spark-2.3.0-Hadoop-2.7-Java-1.8.0_162.eb
 * $CFGS2/s/Spark/Spark-2.4.0-Hadoop-2.7-Java-1.8.eb
 * $CFGS2/s/Spark/Spark-2.4.0-intel-2018b-Hadoop-2.7-Java-1.8-Python-3.6.6.eb
```

So 2.10.0 is available, but that's not the latest one
Launch the build with the provided easyconfig `src/Hadoop-2.10.1-GCCcore-8.3.0-native.eb`

```bash
(node)$> eb src/Hadoop-2.10.1-GCCcore-8.3.0-native.eb -Dr   # Dry-run, check dependencies
(node)$> eb src/Hadoop-2.10.1-GCCcore-8.3.0-native.eb -r
```
Installation will last ~6 minutes using a full `iris` node (`-c 28`).
In general it is preferable to make builds within a screen session.


### Running Hadoop

```
$> module av Hadoop
$> module load tools/Hadoop
```

When doing that, the Hadoop distribution is installed in `$EBROOTHADOOP` (this is set by Easybuild for any loaded software.)

The below instructions are based on the [official tutorial](https://hadoop.apache.org/docs/r2.6.0/hadoop-project-dist/hadoop-common/SingleCluster.html).

#### Hadoop in Single mode

By default, Hadoop is configured to run in a non-distributed mode, as a single Java process. This is useful for debugging.

Let's test it

```bash
$> mkdir -p runs/hadoop/single/input
$> cd runs/hadoop/single
# Prepare input data
$> mkdir input
$> cp ${EBROOTHADOOP}/etc/hadoop/*.xml input
# Map-reduce grep <pattern> -- result is produced in output/
$> hadoop jar ${EBROOTHADOOP}/share/hadoop/mapreduce/hadoop-mapreduce-examples-2.10.1.jar grep input output 'dfs[a-z.]+'
[...]
        File System Counters
                FILE: Number of bytes read=1292924
                FILE: Number of bytes written=3222544
                FILE: Number of read operations=0
                FILE: Number of large read operations=0
                FILE: Number of write operations=0
        Map-Reduce Framework
                Map input records=1
                Map output records=1
                Map output bytes=17
                Map output materialized bytes=25
                Input split bytes=191
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
                GC time elapsed (ms)=0
                Total committed heap usage (bytes)=1029701632
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

#### Pseudo-Distributed Operation

Hadoop can also be run on a single-node in a pseudo-distributed mode where each Hadoop daemon runs in a separate Java process.
Follow the [official tutorial](https://hadoop.apache.org/docs/r2.10.1/hadoop-project-dist/hadoop-common/SingleCluster.html#Pseudo-Distributed_Operation) to ensure you are running in **Single Node Cluster**

Once this is done, follow the [official Wordcount instructions](https://hadoop.apache.org/docs/r2.10.1/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html#Example:_WordCount_v1.0)

#### Full cluster setup

Follow the official instructions of the [Cluster Setup](https://hadoop.apache.org/docs/r2.10.1/hadoop-project-dist/hadoop-common/ClusterSetup.html).

Once this is done, Repeat the execution of the [official Wordcount example](https://hadoop.apache.org/docs/r2.10.1/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html#Example:_WordCount_v1.0).

-----------------------------------------------
## Interactive Big Data Analytics with Spark ##

The objective of this section is to compile and run on [Apache Spark](http://spark.apache.org/)  on top of the [UL HPC](http://hpc.uni.lu) platform.

[Apache Spark](http://spark.apache.org/docs/latest/) is a large-scale data processing engine that performs in-memory computing. Spark offers bindings in Java, Scala, Python and R for building parallel applications.
high-level APIs in Java, Scala, Python and R, and an optimized engine that supports general execution graphs. It also supports a rich set of higher-level tools including Spark SQL for SQL and structured data processing, MLlib for machine learning, GraphX for graph processing, and Spark Streaming.

As for Hadoop, we are first going to build Spark using Easybuild before performing some basic examples. More precisely, in this part, we will review the basic usage of Spark in two cases:

1. a single conffiguration where the classical interactive wrappers (`pyspark`, `scala` and `R` wrappers) will be reviewed.
2. a [Standalone](https://spark.apache.org/docs/latest/spark-standalone.html) cluster configuration - a simple cluster manager included with Spark that makes it easy to set up a cluster), where we will run the Pi estimation.

### Installation

Spark 2.4.3 is available by default (on the 2019b software set) so you can load it.

``` bash
$> module av Spark

------------ /opt/apps/resif/iris/2019b/broadwell/modules/all --------------
   devel/Spark/2.4.3-intel-2019b-Hadoop-2.7-Java-1.8-Python-3.7.4

$> module load devel/Spark
```

You might wish to build and use the most recent version of [Spark](https://spark.apache.org/downloads.html) (_i.e._ at the time of writing 2.4.7 (Dec. 14, 2020) with Pre-built for Apache Hadoop 2.7 or later).
To do that, you will typically have to do the following (not covered in this session by lack of time):

1. Search for the most recent version of Spark provided by Easybuild
    - use the script `scripts/suggest-easyconfigs <pattern>` for that
2. Copy the easyconfig file locally
    - you'll need to get the path to it with `eb -S <pattern>`
3. Rename the file to match the target version
    * Check on the website for the most up-to-date version of the software released
    * Adapt the filename of the copied easyconfig to match the target version / toolchain
        - Ex: `mv Spark-2.4.5-intel-2019b-Python-3.7.4-Java-1.8.eb Spark-2.4.7-intel-2019b-Python-3.7.4-Java-1.8.eb`
4. Edit the content of the easyconfig
   - You'll typically have to adapt the version of the dependencies (use again `scripts/suggest-easyconfigs -s  dep1 dep2 [...]`) and the checksum(s) of the source/patch files to match the static versions set for the target toolchain, enforce https urls etc.


### Interactive usage

Exit your reservation to reload one with the `--exclusive` flag to allocate an exclusive node.
Let's load the installed module:

```bash
(access)$> si -c 28 --exclusive -t 2:00:00
(node)$> source settings/2019
(node)$> module load devel/Spark/2.4.0
```

#### 4.2.a. Pyspark

PySpark is the Spark Python API and exposes Spark Contexts to the Python programming environment.

```bash
$> pyspark
Python 2.7.5 (default, Aug  4 2017, 00:39:18)
[GCC 4.8.5 20150623 (Red Hat 4.8.5-16)] on linux2
Type "help", "copyright", "credits" or "license" for more information.
2018-12-04 13:57:55 WARN  NativeCodeLoader:62 - Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 2.4.0
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
2018-12-04 13:58:43 WARN  NativeCodeLoader:62 - Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
Spark context Web UI available at http://node-1.iris-cluster.uni.lux:4040
Spark context available as 'sc' (master = local[*], app id = local-1543928329271).
Spark session available as 'spark'.
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /___/ .__/\_,_/_/ /_/\_\   version 2.4.0
      /_/

Using Scala version 2.11.12 (Java HotSpot(TM) 64-Bit Server VM, Java 1.8.0_162)
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

Exit (if needed) the previous session.
Ensure that you have connected by SSH to the cluster by opening an SOCKS proxy (see above instructions):

```
(laptop)$> ssh -D 1080 -C iris-cluster
```

Then make a new reservation across multiple nodes:

```bash
# If not yet done, go to the appropriate directory
$> cd ~/git/github.com/ULHPC/tutorials/bigdata

$> srun -p interactive -n 4 -c 7 --exclusive --pty bash
$> mu
$> module av Spark

---------- /home/users/svarrette/.local/easybuild/modules/all ----------
   devel/Spark/2.4.0-Hadoop-2.7-Java-1.8.0_162 (D)

------------ /opt/apps/resif/data/stable/default/modules/all -----------
   devel/Spark/2.3.0-intel-2018a-Hadoop-2.7-Java-1.8.0_162-Python-3.6.4

$> module load devel/Spark/2.4.0
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
$> export MASTER_URL=spark://$(hostname -s):7077   # Helpful
$> echo $MASTER_URL
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

$> start-slave.sh -c ${SLURM_CPUS_PER_TASK} $MASTER_URL
```

Check the result on the master website `http://<IP>:8082`.

Now we can submit an example python Pi estimation script to the Spark cluster with 100 partitions

_Note_: partitions in this context refers of course to Spark's Resilient Distributed Dataset (RDD) and how the dataset is distributed across the nodes in the Spark cluster.

```bash
$> spark-submit --master $MASTER_URL  $SPARK_HOME/examples/src/main/python/pi.py 100
[...]
18/06/13 02:03:43 INFO DAGScheduler: Job 0 finished: reduce at /home/users/svarrette/.local/easybuild/software/devel/Spark/2.2.0-Hadoop-2.6-Java-1.8.0_152/examples/src/main/python/pi.py:43, took 3.738313 s
Pi is roughly 3.140860
```

Finally, at the end, clean your environment and

```bash
# sbin/stop-master.sh - Stops the master that was started via the bin/start-master.sh script.
$SPARK_HOME/sbin/stop-all.sh
```

Prepare a launcher (use your favorite editor) to setup a spark cluster and submit a task to this cluster in batch mode.
Kindly pay attention to the fact that:

* the master is expected to use **1 core** (and 4GiB of RAM) on the first allocated node
    - in particular, the first worker running on the master node will use **1 less core** than  the allocated ones, _i.e._ `$((${SLURM_CPUS_PER_TASK}-1))`
    - once set, the master URL can be obtained with

             MASTER_URL="spark://$(scontrol show hostname $SLURM_NODELIST | head -n 1):7077"

* the workers can use `$SLURM_CPUS_PER_TASK` cores (and a minimum of 1 core)

        export SPARK_WORKER_CORES=${SLURM_CPUS_PER_TASK:-1}

* you can afford 4 GiB per core to the workers, but take into account that Spark master and worker daemons themselves will need 4GiB to run

```bash
  export DAEMON_MEM=${SLURM_MEM_PER_CPU:=4096}
  # Memory to allocate to the Spark master and worker daemons themselves
  export SPARK_DAEMON_MEMORY=${DAEMON_MEM}m
  export SPARK_MEM=$(( ${DAEMON_MEM}*(${SPARK_WORKER_CORES} -1) ))
  # Total amount of memory to allow Spark applications to use on the machine,
  # note that each application's individual memory is configured using its
  # spark.executor.memory property.
  export SPARK_WORKER_MEMORY=${SPARK_MEM}m
  # Options read in YARN client mode
  export SPARK_EXECUTOR_MEMORY=${SPARK_MEM}m
```

_Note_: if you are lazy (or late), you can use the provided launcher script [`runs/launcher.Spark.sh`](https://github.com/ULHPC/tutorials/blob/devel/bigdata/runs/launcher.Spark.sh).

```bash
$> cd runs
$> ./launcher.Spark.sh -h
NAME
  ./launcher.Spark.sh -- Spark Standalone Mode launcher

  This launcher will setup a Spark cluster composed of 1 master and <N> workers,
  where <N> is the number of (full) nodes reserved (i.e. $SLURM_NNODES).
  Then a spark application is submitted (using spark-submit) to the cluster
  By default, $EBROOTSPARK/examples/src/main/python/pi.py is executed.

SYNOPSIS
  ./launcher.Spark.sh -h
  ./launcher.Spark.sh [-i] [path/to/spark_app]

OPTIONS
  -i --interactive
    Interactive mode: setup the cluster and give back the hand
    Only mean with interactive jobs
  -m --master
    Setup a spark master (only)
  -c --client
    Setup spark worker(s)/slave(s). This assumes a master is running
  -n --no-setup
    Do not bootstrap the spark cluster

AUTHOR
  UL HPC Team <hpc-sysadmins@uni.lu>
COPYRIGHT
  This is free software; see the source for copying conditions.  There is
  NO warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

Passive jobs examples:

```bash
############### iris cluster (slurm) ###############
$> sbatch ./launcher.Spark.sh
[...]
```

Once finished, you can check the result of the default application submitted (in `result_${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out`).

```bash
$> cat result_${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out
Pi is roughly 3.141420
```

In case of problems, you can check the logs of the daemons in `~/.spark/logs/`

__Further Reading__

You can find on the Internet many resources for expanding your HPC experience
with Spark. Here are some links you might find useful to go further:

* [Using Spark with GPFS on the ACCRE Cluster](https://bigdata-vandy.github.io/using-spark-with-gpfs/)
* [ARIS notes on Spark](http://doc.aris.grnet.gr/software/spark/)
