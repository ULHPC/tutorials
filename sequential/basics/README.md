[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/sequential/basics/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/sequential/basics/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/sequential/basics/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# HPC Management of Sequential and Embarrassingly Parallel Jobs

     Copyright (c) 2020 S. Varrette and UL HPC Team <hpc-team@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/sequential/basics/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/sequential/basics/slides.pdf)

For many users, the reason to consider (or being encouraged) to offload their computing executions on a (remote) HPC or Cloud facility is tied to the limits reached by their computing devices (laptop or workstation).
It is generally motivated by time constraints:

> "_My computations take several hours/days to complete. On an HPC, it will last a few minutes, no?_"

or search-space explorations:

> "_I need to check my application against a **huge** number of input pieces (files) - it worked on a few of them locally but takes ages for a single check. How to proceed on HPC?_"

For a large majority of you, these questions arise when executing programs that you traditionally run on your laptop or workstation and which consists in:

* a (well-known) application installed on your system, iterated over multiple input conditions configured by specific command-line arguments / configuration files
* a compiled program (C, C++, Java, Go etc.), iterated over multiple input conditions
* your favorite R or python (custom) development scripts, iterated again over multiple input conditions

**Be aware that in most of the cases, these applications are inherently SERIAL**: These are **able to use only one core** when executed.
You thus deal with what is often call a _Bag of (independent) tasks_, also referred to as **embarrassingly parallel tasks**.
The objective of this practical session is to guide you toward the effective management of such tasks in a way that optimize both the resource allocation and the completion time.


--------------------
## Pre-requisites ##

Ensure you are able to [connect to the UL HPC clusters](https://hpc.uni.lu/users/docs/access.html).
In particular, recall that the `module` command **is not** available on the access frontends.

If you have never configured [GNU Screen](http://www.gnu.org/software/screen/) before, and while not strictly mandatory, we advise you to rely on our customized configuration file for screen [`.screenrc`](https://github.com/ULHPC/dotfiles/blob/master/screen/.screenrc) available on [Github](https://github.com/ULHPC/dotfiles/blob/master/screen/.screenrc) and available on the access nodes under `/etc/dotfiles.d/screen/.screenrc`

```bash
### Access to ULHPC cluster - here iris
(laptop)$> ssh iris-cluster
# /!\ Advanced (but recommended) best-practice:
#    always work within an GNU Screen session named with 'screen -S <topic>' (Adapt accordingly)
# IIF not yet done, copy ULHPC .screenrc in your home
(access)$> cp /etc/dotfiles.d/screen/.screenrc ~/
```

Now you'll need to pull the latest changes in your working copy of the [ULHPC/tutorials](https://github.com/ULHPC/tutorials) you should have cloned in `~/git/github.com/ULHPC/tutorials` (see ["preliminaries" tutorial](../../preliminaries/))

``` bash
(access)$> cd ~/git/github.com/ULHPC/tutorials
(access)$> git pull
```

Now **configure a dedicated directory `~/tutorials/sequential` for this session**

``` bash
# return to your home
(access)$> mkdir -p ~/tutorials/sequential
(access)$> cd ~/tutorials/sequential
# create a symbolic link to the reference material
(access)$> ln -s ~/git/github.com/ULHPC/tutorials/sequential/basics ref.d
```

**Advanced users** (_eventually_ yet __strongly__ recommended), create a [GNU Screen](http://www.gnu.org/software/screen/) session you can recover later - see ["Getting Started" tutorial](../../getting-started/)

``` bash
# /!\ Advanced (but recommended) best-practice:
#     Always work within an GNU Screen session named with 'screen -S <topic>' (Adapt accordingly)
#     _note_: assumes you have previously copied ULHPC .screenrc in your homedir
(access-iris)$> screen -S HPC-school
#    CTRL + a c: (create) creates a new Screen window. The default Screen number is zero.
#    CTRL + a n: (next) switches to the next window.
#    CTRL + a p: (prev) switches to the previous window.
#    CTRL + a A: (title) rename the current window
#    CTRL + a d: (detach) detaches from a Screen -
# Once detached:
#   screen -ls : list available screen
#   screen -x    reattach to a past screen
```

## A sample "Stress Me!" parameter exploration

In this sample scenario, we will mimic the exploration of integer parameters within a continuous range `[1..n]` for serial (1-core) tasks lasting different amount time (here ranging ranging from 1 to n seconds). This is very typical: your serial application is likely to take more or less time depending on its execution context (typically controlled by command line arguments).
To keep it simple, we will assume the n=30 by default.

You will use a script, `scripts/run_stressme`, to reproduce a configurable amount of  stress on the system set for a certain amount of time (by default: 20s) passed as parameter using the '[stress](https://linux.die.net/man/1/stress)' command.
We would like thus to conduct the following executions:

``` bash
run_stressme 1     # execution time: 1s
run_stressme 2     # execution time: 2s
run_stressme 3     # execution time: 3s
run_stressme 4     # execution time: 4s
run_stressme 5     # execution time: 5s
run_stressme 6     # execution time: 6s
run_stressme 7     # execution time: 7s
[...]
run_stressme 27    # execution time: 27s
run_stressme 28    # execution time: 28s
run_stressme 29    # execution time: 29s
run_stressme 30    # execution time: 30s
```

Running this job campaign **sequentially** (one after the other on the same core) would take approximatively **465s**.
The below table wrap up the sequential times required for the completion of the job campaign depending on the number of `run_stressme` tasks, and the _theoretical_ optimal time corresponding to the _slowest_ task to complete when using an infinite number of computing resources.

| n = Number of tasks | Expected Seq. time to complete | Optimal time |
|---------------------|--------------------------------|--------------|
| 1                   | 1s                             | 1s           |
| 10                  | 55s      (~1 min)              | 10s          |
| __30 (default)__    | __465s     (7 min 45s)__       | __30s__      |
| 100                 | 5050s    (1h 24 min 10s)       | 100s         |


The objective is of course to optimize the time required to conclude this exploration


### Single task run (interactive)

Let's first test this command in an interactive jobs:

```bash
### Access to ULHPC cluster (if not yet done)
(laptop)$> ssh iris-cluster
### Have an interactive job
# ... either directly
(access)$> si
# ... or using the HPC School reservation 'hpcschool'if needed  - use 'sinfo -T' to check if active and its name
# (access)$> srun --reservation=hpcschool --pty bash
(node)$>
```

In another terminal (or another screen tab/windows), connect to that job and run `htop`

``` bash
# check your running job
(access)$> sq
# squeue -u $(whoami)
   JOBID PARTIT       QOS                 NAME       USER NODE  CPUS ST         TIME    TIME_LEFT PRIORITY NODELIST(REASON)
 2171206  [...]
# Connect to your running job, identified by its Job ID
(access)$> sjoin 2171206     # /!\ ADAPT job ID accordingly, use <TAB> to have it autocatically completed
(node)$> htop # view of all processes
#               F5: tree view
#               u <name>: filter by process of <name>
#               q: quit
```
Note that you have **only one core** reserved by default in a interactive job: you may thus see activities on the other cores via `htop`, they are corresponding to processes run by the other users. Note that you can use the `--exclusive`  flag upon reservation to request en exclusive access to a full node.

Now execute the `run_stressme` command in the first terminal/windows to see its effect on the processor load:

``` bash
# Run directly
(node)$> ~/git/github.com/ULHPC/tutorials/sequential/basics/scripts/run_stressme
#  /usr/bin/stress -c 1 -t 20
stress: info: [59918] dispatching hogs: 1 cpu, 0 io, 0 vm, 0 hdd
stress: info: [59918] successful run completed in 20s

# Now let's create a convenient path to the scripts under the dedicated directory
# used for this session
(node)$> cd ~/tutorials/sequential
# create a symbolic link to the script directory
(node)$> ln -s ref.d/scripts .    # <-- don't forget the trailing '.'  means 'here'
(node)$> ls scripts/run_stressme      # should not fail
scripts/run_stressme
(node)$> ./scripts/run_stressme
stress: info: [59918] dispatching hogs: 1 cpu, 0 io, 0 vm, 0 hdd
stress: info: [59918] successful run completed in 20s
# Exit the job
(node)$> exit    # OR CTRL-D
(access)$>
```

Exit `htop` in its terminal (press 'q' to exit) and press `CTRL-D` to disconnect to return to the access server.
Quit the interactive job by pressing `CTRL-D` to disconnect to return to the access server.


## A First launcher (1 job, 1 task on one core)

We will create a first "_launcher_ script", responsible to perform a single task over 1 core.
We will copy the default template provided by the ULHPC team for such usage, [`launcher.serial.sh`](scripts/launcher.serial.sh).

```bash
(access)$> cd ~/tutorials/sequential # if not yet done
(access)$> cp ~/git/github.com/ULHPC/tutorials/sequential/basics/scripts/launcher.serial.sh .   # <- trailing '.' means 'here
(access)$> mv launcher.serial.sh launcher.stressme-serial.sh
```

Use your favorite editor (`nano`, `vim` etc) to edit it as follows:

```diff
--- ~/git/github.com/ULHPC/tutorials/sequential/basics/scripts/launcher.stressme-serial.sh 2020-12-10 16:25:24.564580000 +0100
+++ launcher.stressme-serial.sh 2020-12-10 17:02:16.847752000 +0100
 #! /bin/bash -l
 ############################################################################
 # Default launcher for serial (one core) tasks
 ############################################################################
-###SBATCH -J Serial-jobname
+#SBATCH -J StressMe-single
@@ -32,7 +32,7 @@
 # /!\ ADAPT TASK variable accordingly
 # Absolute path to the (serial) task to be executed i.e. your favorite
 # Java/C/C++/Ruby/Perl/Python/R/whatever program to be run
-TASK=${TASK:=${HOME}/bin/app.exe}
+TASK=${TASK:=${HOME}/tutorials/sequential/scripts/run_stressme}
```

Let's test it in an interactive job:

```bash
### Have an interactive job
# ... either directly
(access)$> si
# ... or using the HPC School reservation 'hpcschool'if needed  - use 'sinfo -T' to check if active and its name
# (access)$> srun --reservation=hpcschool --pty bash

# check usage
(node)$> ./launcher.stressme-serial.sh -h

### DRY-RUN
(node)$> ./launcher.stressme-serial.sh -n
### Starting timestamp (s): 1607616148
/home/users/<login>/tutorials/sequential/scripts/run_stressme
### Ending timestamp (s): 1607616148"
# Elapsed time (s): 0

### REAL test
(node)$> ./launcher.stressme-serial.sh
### Starting timestamp (s): 1607616167
#  /usr/bin/stress -c 1 -t 20
stress: info: [64361] dispatching hogs: 1 cpu, 0 io, 0 vm, 0 hdd
stress: info: [64361] successful run completed in 20s
### Ending timestamp (s): 1607616187"
# Elapsed time (s): 20
```

_Hint_: for the lazy persons, you can define on the fly the `TASK` variable as follows:

```bash
### DRY-RUN
(node)$> TASK=$(pwd)/scripts/run_stressme ./scripts/launcher.serial.sh -n
### Starting timestamp (s): 1607616148
/home/users/<login>/tutorials/sequential/scripts/run_stressme
### Ending timestamp (s): 1607616148"
# Elapsed time (s): 0

### REAL test
(node)$> TASK=$(pwd)/scripts/run_stressme ./scripts/launcher.serial.sh
### Starting timestamp (s): 1607616500
#  /usr/bin/stress -c 1 -t 20
stress: info: [103707] dispatching hogs: 1 cpu, 0 io, 0 vm, 0 hdd
stress: info: [103707] successful run completed in 20s
### Ending timestamp (s): 1607616520"
# Elapsed time (s): 20
```

Quit your interactive job (exit or `CTRL+D`) and submit it as a passive job:

``` bash
# Note: you may want/need to run it under the dedicated reservation set for the training event
# using sbatch --reservation=[...]
(access)$> sbatch ./launcher.stressme-serial.sh
```

_Hint_: still for the lazy persons, just run:

``` bash
# Note: you may want/need to run it under the dedicated reservation set for the training event
# using [...] sbatch --reservation=[...]
#    TASK=$(pwd)/scripts/run_stressme  sbatch ./scripts/launcher.serial.sh
```

### A VERY BAD StressMe Job Campaign: For loop on  `sbatch`

**Now the job campaign for N tasks (N=30) can be obtained by submitting N=30 jobs**:

``` bash
# Note: you may want/need to run it under the dedicated reservation set for the training event
# using sbatch --reservation=[...]
# /!\ ALWAYS echo your commands in a for loop before going real
(access)$> for i in $(seq 1 30); do echo sbatch ./launcher.stressme-serial.sh $i; done
sbatch ./launcher.stressme-serial.sh 1
sbatch ./launcher.stressme-serial.sh 2
sbatch ./launcher.stressme-serial.sh 3
[...]
sbatch ./launcher.stressme-serial.sh 28
sbatch ./launcher.stressme-serial.sh 29
sbatch ./launcher.stressme-serial.sh 30
# Equivalent of
#   for i in {1..30}; do echo sbatch ./launcher.stressme-serial.sh $i; done

# Go real
(access)$> for i in $(seq 1 30); do sbatch ./launcher.stressme-serial.sh $i; done
Submitted batch job 2171802
Submitted batch job 2171803
Submitted batch job 2171804
Submitted batch job 2171805
Submitted batch job 2171806
Submitted batch job 2171807
Submitted batch job 2171809
Submitted batch job 2171810
Submitted batch job 2171811
Submitted batch job 2171812
Submitted batch job 2171813
Submitted batch job 2171814
Submitted batch job 2171815
Submitted batch job 2171816
Submitted batch job 2171817
Submitted batch job 2171818
Submitted batch job 2171819
Submitted batch job 2171820
Submitted batch job 2171821
Submitted batch job 2171822
Submitted batch job 2171823
Submitted batch job 2171824
Submitted batch job 2171825
Submitted batch job 2171827
Submitted batch job 2171828
Submitted batch job 2171829
Submitted batch job 2171830
Submitted batch job 2171831
Submitted batch job 2171832
Submitted batch job 2171834
```

Use `sq` to check the status of your job in the queue:

``` bash
(access)$> sq   # squeue -u $(whoami)
   JOBID PARTIT       QOS                 NAME       USER NODE  CPUS ST         TIME    TIME_LEFT PRIORITY NODELIST(REASON)
 2171819  batch    normal      StressMe-single  <login>    1     1 CG         0:19        59:41    12678 iris-081
 2171820  batch    normal      StressMe-single  <login>    1     1 CG         0:19        59:41    12678 iris-149
 2171818  batch    normal      StressMe-single  <login>    1     1 CG         0:18        59:42    12678 iris-081
 2171831  batch    normal      StressMe-single  <login>    1     1  R         0:13        59:47    12678 iris-075
 2171832  batch    normal      StressMe-single  <login>    1     1  R         0:13        59:47    12678 iris-075
 2171834  batch    normal      StressMe-single  <login>    1     1  R         0:13        59:47    12678 iris-075
 2171830  batch    normal      StressMe-single  <login>    1     1  R         0:15        59:45    12678 iris-075
 2171825  batch    normal      StressMe-single  <login>    1     1  R         0:16        59:44    12678 iris-082
 2171827  batch    normal      StressMe-single  <login>    1     1  R         0:16        59:44    12678 iris-082
 2171828  batch    normal      StressMe-single  <login>    1     1  R         0:16        59:44    12678 iris-082
 2171829  batch    normal      StressMe-single  <login>    1     1  R         0:16        59:44    12678 iris-082
 2171823  batch    normal      StressMe-single  <login>    1     1  R         0:18        59:42    12678 iris-149
 2171824  batch    normal      StressMe-single  <login>    1     1  R         0:18        59:42    12678 iris-149
 2171821  batch    normal      StressMe-single  <login>    1     1  R         0:19        59:41    12678 iris-149
 2171822  batch    normal      StressMe-single  <login>    1     1  R         0:19        59:41    12678 iris-149
```

Check the past jobs statistics upon completion:

``` bash
(access)$> sacct -X --format User,JobID,partition%12,state,time,start,end,elapsed,nnodes,ncpus,nodelist
   User        JobID    Partition      State  Timelimit               Start                 End    Elapsed   NNodes      NCPUS        NodeList
------- ------------ ------------ ---------- ---------- ------------------- ------------------- ---------- -------- ---------- ---------------
<login> 2171802             batch  COMPLETED   01:00:00 2020-12-10T17:42:38 2020-12-10T17:42:39   00:00:01        1          1        iris-075
<login> 2171803             batch  COMPLETED   01:00:00 2020-12-10T17:42:41 2020-12-10T17:42:43   00:00:02        1          1        iris-075
<login> 2171804             batch  COMPLETED   01:00:00 2020-12-10T17:42:41 2020-12-10T17:42:44   00:00:03        1          1        iris-075
<login> 2171805             batch  COMPLETED   01:00:00 2020-12-10T17:42:41 2020-12-10T17:42:45   00:00:04        1          1        iris-075
<login> 2171806             batch  COMPLETED   01:00:00 2020-12-10T17:42:41 2020-12-10T17:42:47   00:00:06        1          1        iris-075
<login> 2171807             batch  COMPLETED   01:00:00 2020-12-10T17:42:41 2020-12-10T17:42:48   00:00:07        1          1        iris-075
<login> 2171809             batch  COMPLETED   01:00:00 2020-12-10T17:42:41 2020-12-10T17:42:49   00:00:08        1          1        iris-075
<login> 2171810             batch  COMPLETED   01:00:00 2020-12-10T17:42:41 2020-12-10T17:42:50   00:00:09        1          1        iris-075
<login> 2171811             batch  COMPLETED   01:00:00 2020-12-10T17:42:44 2020-12-10T17:42:54   00:00:10        1          1        iris-076
<login> 2171812             batch  COMPLETED   01:00:00 2020-12-10T17:42:44 2020-12-10T17:42:55   00:00:11        1          1        iris-076
<login> 2171813             batch  COMPLETED   01:00:00 2020-12-10T17:42:44 2020-12-10T17:42:56   00:00:12        1          1        iris-076
<login> 2171814             batch  COMPLETED   01:00:00 2020-12-10T17:42:44 2020-12-10T17:42:57   00:00:13        1          1        iris-076
<login> 2171815             batch  COMPLETED   01:00:00 2020-12-10T17:42:44 2020-12-10T17:42:58   00:00:14        1          1        iris-076
<login> 2171816             batch  COMPLETED   01:00:00 2020-12-10T17:42:45 2020-12-10T17:42:59   00:00:14        1          1        iris-081
<login> 2171817             batch  COMPLETED   01:00:00 2020-12-10T17:42:46 2020-12-10T17:43:01   00:00:15        1          1        iris-081
<login> 2171818             batch  COMPLETED   01:00:00 2020-12-10T17:42:46 2020-12-10T17:43:04   00:00:18        1          1        iris-081
<login> 2171819             batch  COMPLETED   01:00:00 2020-12-10T17:42:47 2020-12-10T17:43:06   00:00:19        1          1        iris-081
<login> 2171820             batch  COMPLETED   01:00:00 2020-12-10T17:42:47 2020-12-10T17:43:06   00:00:19        1          1        iris-149
<login> 2171821             batch  COMPLETED   01:00:00 2020-12-10T17:42:48 2020-12-10T17:43:09   00:00:21        1          1        iris-149
<login> 2171822             batch  COMPLETED   01:00:00 2020-12-10T17:42:48 2020-12-10T17:43:09   00:00:21        1          1        iris-149
<login> 2171823             batch  COMPLETED   01:00:00 2020-12-10T17:42:49 2020-12-10T17:43:11   00:00:22        1          1        iris-149
<login> 2171824             batch  COMPLETED   01:00:00 2020-12-10T17:42:49 2020-12-10T17:43:12   00:00:23        1          1        iris-149
<login> 2171825             batch  COMPLETED   01:00:00 2020-12-10T17:42:51 2020-12-10T17:43:14   00:00:23        1          1        iris-082
<login> 2171827             batch  COMPLETED   01:00:00 2020-12-10T17:42:51 2020-12-10T17:43:17   00:00:26        1          1        iris-082
<login> 2171828             batch  COMPLETED   01:00:00 2020-12-10T17:42:51 2020-12-10T17:43:17   00:00:26        1          1        iris-082
<login> 2171829             batch  COMPLETED   01:00:00 2020-12-10T17:42:51 2020-12-10T17:43:19   00:00:28        1          1        iris-082
<login> 2171830             batch  COMPLETED   01:00:00 2020-12-10T17:42:52 2020-12-10T17:43:19   00:00:27        1          1        iris-075
<login> 2171831             batch  COMPLETED   01:00:00 2020-12-10T17:42:54 2020-12-10T17:43:22   00:00:28        1          1        iris-075
<login> 2171832             batch  COMPLETED   01:00:00 2020-12-10T17:42:54 2020-12-10T17:43:23   00:00:29        1          1        iris-075
<login> 2171834             batch  COMPLETED   01:00:00 2020-12-10T17:42:54 2020-12-10T17:43:24   00:00:30        1          1        iris-075
```

As can be seen, the campaign was initiated at 17:42:38 to be completed for the latest job at 17:43:24: **it took thus 46s to be completed**. It's **90,1% improvement on the sequential completion time**.
Note that you can quickly get the maximum value of the 7th column with `sort -k 7 [-n]` as follows:

``` bash
(access)$> sacct -X --format User,JobID,partition%12,state,time,start,end,elapsed,nnodes,ncpus,nodelist --noheader | sort -k 7 | tail -n 2
```
Repeating the experience on **100** consecutive tests demonstrated a completion between 18:43:22 and 18:46:19, i.e. on **less than 3 minutes (2m 57s or 177s)**. It corresponds to a **96,5% improvement on the sequential completion time**.
The below table summarizes the results:

| #tasks | #jobs | Seq. time | Optimal | `for [...] sbatch [...]` | Seq. Time Decrease | #nodes         |
|--------|-------|-----------|---------|--------------------------|--------------------|----------------|
| 1      | 1     | 1s        | 1s      | 1s                       | 0%                 | 1 (max 1)      |
| 10     | 10    | 55s       | 10s     | 19s                      | 65%                | 3 (max 10)     |
| __30__ | 30    | __465s__  | __30s__ | __46s__                  | __90,1% __         | __5 (max 30)__ |
| 100    | 100s  | 5050s     | 100s    | 177s                     | 96,5%              | 10  (max 100)  |


This works of course but this is **generally against best-practices**:

* **to complete N (serial) tasks, you need to submit N jobs that could be spread on up to N different nodes**.
    - This induces an **non-necessary overload of the scheduler** for (generally) very short tasks
* Node coverage is sub-optimal
    - your serial jobs can be spread on **up to N different nodes**

Imagine expanding the job campaign to 1000 or 10000 test cases if not more, you risk to degrade significantly the HPC environment (the scheduler will likely have trouble to manage so many short-live jobs).

To prevent this behaviour, **We have thus limit the number of jobs allowed per user**  (see `sqos`).
The objective is to invite you _gently_ to **regroup your tasks per node** in order to better exploit their many-core configuration (28 cores on `iris`, 128 on `aion`).


### A BAD StressMe Job Campaign: job arrays

Slurm support [Job Arrays](https://slurm.schedmd.com/job_array.html), a mechanism for submitting and managing collections of similar jobs quickly and easily;
All jobs must have the same initial options (e.g. size, time limit, etc.) and you can limit the numner
You just need to specify the array index values using the `--array` or `-a` option of the `sbatch` command. You **SHOULD** set the maximum number of simultaneously running tasks from the job array  using a "%" separator -- typically match the number of cores per node (28 on `iris`, 128 on `aion`). For example `--array=0-100%28" will limit the number of simultaneously running tasks from this job array to 28.
Job arrays will have several additional environment variable set:

* `$SLURM_ARRAY_JOB_ID`     will be set to the first job ID of the array
* `$SLURM_ARRAY_TASK_ID`    will be set to the job array index value.
* `$SLURM_ARRAY_TASK_COUNT` will be set to the number of tasks in the job array.

We will copy the previous launcher:

```bash
(access)$> cd ~/tutorials/sequential # if not yet done
(access)$> cp launcher.stressme-serial.sh launcher.stressme-jobarray.sh
```

Use your favorite editor (`nano`, `vim` etc) to edit it as follows:

```diff
@@ -14,7 +14,8 @@
 #SBATCH --ntasks-per-node 1
 #SBATCH -c 1                   # multithreading per task : -c --cpus-per-task <n> request
 #__________________________
-#SBATCH -o logs/%x-%j.out      # log goes into logs/<jobname>-<jobid>.out
+#SBATCH --array 1-9%5
+#SBATCH -o logs/%x-%A_%a.out   # log goes into logs/<jobname>-<masterID>_<taskID>.out
@@ -75,7 +74,7 @@
 start=$(date +%s)
 echo "### Starting timestamp (s): ${start}"

-${CMD_PREFIX} ${TASK} ${OPTS}
+${CMD_PREFIX} ${TASK} ${OPTS} ${SLURM_ARRAY_TASK_ID}
```

Now you can run the same job campaign as before with a single `sbatch` command:

``` bash
(access)$> sbatch --array 1-30%28 ./launcher.stressme-jobarray.sh
```

**The above command WON'T work**: massive job arrays campaign were run in the past that used to overwhelm the slurm controllers.
To avoid this behaviour to repeat, **we drastically reduce the capabilities of job arrays**:

``` bash
(access)$> scontrol show config | grep -i maxarray
```

_In short_, **Don't use job arrays!!!**: you can do better with [GNU Parallel](http://www.gnu.org/software/parallel/)

----------------------------------------------------
## A better launcher (1 job, #cores tasks per node)

As advised, you are encouraged to aggregate several serial tasks within a single job to better exploit the many-core configuration of each nodes (28 or 128 cores per node) - this would clearly be beneficial compared to you laptop that probably have "_only_" 4 cores.

One natural way of doing so would be to aggregate these tasks within a single slurm launcher, start them in the background (i.e. as child processes) by using the ampersand `&` after a Bash command, and the `wait` command:

``` bash
# Dangerous
TASK=run_stressme
for i in {1..30}; do
    srun -n1 --exclusive -c 1 --cpu-bind=cores ${TASK} $i &
done
wait
```
The ampersand `&` spawns the command `srun -n1 --exclusive -c 1 --cpu-bind=cores ${TASK} $i`  in the background and allows the loop to continue to the next iteration without waiting for this sub-process to finish.
This approach is dangerous and the location of the `wait` command can be optimized to match the number of tasks per nodes as ideally, you would target to execute your bag of tasks by groups of 28 (or 128) to match the hardware configuration of ULHPC nodes.
In generic terms, you wish to target `${SLURM_NTASKS_PER_NODE}` (if set) or the output of `nproc --all` (28 on iris) process to be forked assuming you use a full node.
The code would be as follows:

``` bash
# Better but still dangerous
TASK=run_stressme
ncores=${SLURM_NTASKS_PER_NODE:-$(nproc --all)}
For i in {1..30}; do
    srun -n1 --exclusive -c 1 --cpu-bind=cores ${TASK} $i &
    [[ $((i%ncores)) -eq 0 ]] && wait
done
wait
```

You can test it from the sample launcher [`launcher.serial-ampersand.sh`](scripts/launcher.serial-ampersand.sh).

```bash
(access)$> cd ~/tutorials/sequential # if not yet done
(access)$> cp ~/git/github.com/ULHPC/tutorials/sequential/basics/scripts/launcher.serial-ampersand.sh .   # <- trailing '.' means 'here
(access)$> mv launcher.serial-ampersand.sh launcher.stressme-serial-ampersand.sh
```

Use your favorite editor (`nano`, `vim` etc) to edit it as follows:

```diff
--- scripts/launcher.serial-ampersand.sh        2020-12-12 17:05:02.346701013 +0100
+++ launcher.stressme-serial-ampersand.sh       2020-12-12 17:06:30.001896000 +0100
@@ -5,7 +5,7 @@
 # within one node using the Bash & (ampersand), a builtin control operator
 # used to fork processes, and the wait command.
 ############################################################################
-###SBATCH -J Serial-ampersand
+#SBATCH -J StressMe-ampersand
 #SBATCH --time=0-01:00:00      # 1 hour
 #SBATCH --partition=batch
 #__________________________
@@ -24,7 +24,7 @@
 # /!\ ADAPT TASK variable accordingly
 # Absolute path to the (serial) task to be executed i.e. your favorite
 # Java/C/C++/Ruby/Perl/Python/R/whatever program to be run
-TASK=${TASK:=${HOME}/bin/app.exe}
+TASK=${TASK:=${HOME}/tutorials/sequential/scripts/run_stressme}
 MIN=1
 MAX=30
 NCORES=${SLURM_NTASKS_PER_NODE:-$(nproc --all)}
```

Now launch the job campaign as before with a single `sbatch` command:


``` bash
# Note: you may want/need to run it under the dedicated reservation set for the training event
# using sbatch --reservation=[...]
(access)$> sbatch ./launcher.stressme-serial-ampersand.sh
Submitted batch job 2175666
# if you have not done it upon submission, you can correct it with (adapt accordingly):
#     scontrol update jobid=<JOBID> reservationname=<name>
```

_Hint_: still for the lazy persons afraid of copies/modifications, just run:

``` bash
# Note: you may want/need to run it under the dedicated reservation set for the training event
# using [...] sbatch --reservation=[...]
#       TASK=$(pwd)/scripts/run_stressme sbatch ./scripts/launcher.serial-ampersand.sh
```

Check the duration of the job afterward:
``` bash
# See all job steps with slist <jobID>
(access)$> slist 2175666
# sacct -j 2175666 --format User,JobID,Jobname%30,partition,state,time,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist,AveCPU,ConsumedEnergyRaw
     User        JobID                        JobName  Partition      State  Timelimit    Elapsed     MaxRSS  MaxVMSize   NNodes      NCPUS        NodeList     AveCPU ConsumedEnergyRaw
--------- ------------ ------------------------------ ---------- ---------- ---------- ---------- ---------- ---------- -------- ---------- --------------- ---------- -----------------
svarrette 2175666                  StressMe-ampersand      batch  COMPLETED   01:00:00   00:00:59                              1         28        iris-121                        11885
          2175666.bat+                          batch             COMPLETED              00:00:59      7472K    178788K        1         28        iris-121   00:00:00             11850
          2175666.ext+                         extern             COMPLETED              00:00:59          0    108056K        1         28        iris-121   00:00:00             11885
          2175666.0                      run_stressme             COMPLETED              00:00:03          0    248544K        1          1        iris-121   00:00:00               949
          2175666.1                      run_stressme             COMPLETED              00:00:04          0    248544K        1          1        iris-121   00:00:00              1279
          2175666.2                      run_stressme             COMPLETED              00:00:01          0    248544K        1          1        iris-121   00:00:00               276
          2175666.3                      run_stressme             COMPLETED              00:00:07          0    248544K        1          1        iris-121   00:00:00              1952
          2175666.4                      run_stressme             COMPLETED              00:00:08          0    248544K        1          1        iris-121   00:00:00              2526
          2175666.5                      run_stressme             COMPLETED              00:00:02          0    248544K        1          1        iris-121   00:00:00               635
[...]
          2175666.22                     run_stressme             COMPLETED              00:00:20          0    248544K        1          1        iris-121   00:00:00              5395
          2175666.23                     run_stressme             COMPLETED              00:00:25          0    248544K        1          1        iris-121   00:00:00              6434
          2175666.24                     run_stressme             COMPLETED              00:00:18          0    248544K        1          1        iris-121   00:00:00              4930
          2175666.25                     run_stressme             COMPLETED              00:00:23          0    248544K        1          1        iris-121   00:00:00              6027
          2175666.26                     run_stressme             COMPLETED              00:00:24          0    248544K        1          1        iris-121   00:00:00              6228
          2175666.27                     run_stressme             COMPLETED              00:00:28          0    248544K        1          1        iris-121   00:00:00              6944
          2175666.28                     run_stressme             COMPLETED              00:00:30          0    248544K        1          1        iris-121   00:00:00              4413
          2175666.29                     run_stressme             COMPLETED              00:00:30       392K    248544K        1          1        iris-121   00:00:29              4554
#
# seff 2175666
#
Job ID: 2175666
Cluster: iris
User/Group: svarrette/clusterusers
State: COMPLETED (exit code 0)
Nodes: 1
Cores per node: 28
CPU Utilized: 00:07:45
CPU Efficiency: 28.15% of 00:27:32 core-walltime
Job Wall-clock time: 00:00:59
Memory Utilized: 7.30 MB
Memory Efficiency: 0.01% of 112.00 GB
```

You can get the aggregated statistics with `-X`:

``` bash
# See aggregated statictis with slist <jobID> -X
(access)$> slist 2175666 -X
# sacct -j 2175666 --format User,JobID,Jobname%30,partition,state,time,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist,AveCPU,ConsumedEnergyRaw -X
     User        JobID                        JobName  Partition      State  Timelimit    Elapsed     MaxRSS  MaxVMSize   NNodes      NCPUS        NodeList     AveCPU ConsumedEnergyRaw
--------- ------------ ------------------------------ ---------- ---------- ---------- ---------- ---------- ---------- -------- ---------- --------------- ---------- -----------------
svarrette 2175666                  StressMe-ampersand      batch  COMPLETED   01:00:00   00:00:59                              1         28        iris-121                        11885
# [...]
```
The launcher script supports the option `-c` to limit the number of cores used.
Check it out it's effect on the `htop` terminal/session:

``` bash
# Note: you may want/need to run it under the dedicated reservation set for the training event
# using sbatch --reservation=[...]
(access)$> sbatch ./launcher.stressme-serial-ampersand.sh -c 14
```

Finally, repeat with 100 and 10 tasks (aggreated within a single job):

``` bash
# Note: you may want/need to run it under the dedicated reservation set for the training event
# using sbatch --reservation=[...]
(access)$> sbatch ./launcher.stressme-serial-ampersand.sh --max 100
Submitted batch job 2175664
(access)$> sbatch ./launcher.stressme-serial-ampersand.sh --max 10
Submitted batch job 2175665
# sq [...]
# Once completed, check all statistics - # /!\ ADAPT JobID accordingly
(access)$> slist 2175664,2175665,2175666 -X
# sacct -j 2175664,2175665,2175666 --format User,JobID,Jobname%30,partition,state,time,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist,AveCPU,ConsumedEnergyRaw -X
User        JobID                        JobName  Partition      State  Timelimit    Elapsed     MaxRSS  MaxVMSize   NNodes      NCPUS        NodeList     AveCPU ConsumedEnergyRaw
--------- ------------ ------------------------------ ---------- ---------- ---------- ---------- ---------- ---------- -------- ---------- --------------- ---------- -----------------
svarrette 2175664                  StressMe-ampersand      batch  COMPLETED   01:00:00   00:04:33                              1         28        iris-117                        76260
svarrette 2175665                  StressMe-ampersand      batch  COMPLETED   01:00:00   00:00:11                              1         28        iris-121                         1866
svarrette 2175666                  StressMe-ampersand      batch  COMPLETED   01:00:00   00:00:59                              1         28        iris-121                        11885
```

Compared to the (**VERY BAD**) "for loop of sbatch" approach, we have thus drastically optimized the resources allocation (a single node allocated, when 10 and up to 100 were required for the bigger campaign (100 tasks)) at the price of a affordable time penalty, still largely beneficial when compared to sequential i.e.:

* for 30 tasks:  28% penalty (59s  vs. 46s), still demonstrating a 87% improvement compared to the sequential run.
* for 100 tasks: 54% penalty (273s vs 177s), still demonstrating a 95% improvement compared to the sequential run.

Nevertheless, we had to revisit the logic of the launcher script and enforce several customization.
On the contrary, [GNU Parallel](http://www.gnu.org/software/parallel/) allow flexibility and adaptability while minimizing the customization: it makes the pallalelization happen automagically based on the slurm constraits (`--ntasks-per-node`, `-c`).

We will focus on a **single node run** for reasons make clear later.


----------------------------------------------------------------
## Best launcher based on GNU Parallel (1 job, 1 node, N tasks)

![](https://www.gnu.org/software/parallel/logo-gray+black300.png)

[GNU Parallel](http://www.gnu.org/software/parallel/) is a tool for executing tasks in parallel, typically on a single machine. When coupled with the Slurm command `srun`, parallel becomes a powerful way of distributing a set of embarrassingly parallel tasks amongst a number of workers. This is particularly useful when the number of tasks is significantly larger than the number of available workers (i.e. `$SLURM_NTASKS`), and each tasks is independent of the others.

**Follow now our [GNU Parallel tutorial](../gnu-parallel/) to become more accustomed with the special (complex) syntax of this tool.**

To illustrate the advantages of this approach, we will use the generic GNU parallel launcher script designed by the UL HPC Team **[`scripts/launcher.parallel.sh`](https://github.com/ULHPC/tutorials/blob/devel/sequential/basics/scripts/launcher.parallel.sh)**.
First copy this script template and **make it your FINAL launcher for stressme**:

```bash
(access)$> cd ~/tutorials/sequential # if not yet done
(access)$> cp ~/git/github.com/ULHPC/tutorials/sequential/basics/scripts/launcher.parallel.sh .   # <- trailing '.' means 'here
(access)$> mv launcher.parallel.sh launcher.stressme.sh
```

Use your favorite editor (`nano`, `vim` etc) to edit it as follows:

```diff
--- scripts/launcher.parallel.sh   2020-12-12 14:28:56.978642778 +0100
+++ launcher.stressme.sh           2020-12-12 14:31:14.432175130 +0100
@@ -55,9 +55,9 @@
 ### /!\ ADAPT TASK and TASKLIST[FILE] variables accordingly
 # Absolute path to the (serial) task to be executed i.e. your favorite
 # Java/C/C++/Ruby/Perl/Python/R/whatever program to be run
-TASK=${TASK:=${HOME}/bin/app.exe}
+TASK=${TASK:=${HOME}/tutorials/sequential/scripts/run_stressme}
 # Default task list, here {1..8} is a shell expansion interpreted as 1 2... 8
-TASKLIST="{1..8}"
+TASKLIST="{1..30}"
 # TASKLISTFILE=path/to/input_file
 # If non-empty, used as input source over TASKLIST.
 TASKLISTFILE=
```

So not far from what you did form the basic first launcher.

### Interactive test of ULHPC GNU Parallel launcher (exclusive job required)

You can test this launcher within an **exclusive** interactive job (otherwise the internal calls to `srun --exclusive` will conflict with the default settings of interactive jobs)

``` bash
### Have an **exclusive** interactive job for 4 (embarrassingly parallel) tasks
# DON'T forget the exclusive flag for interactive tests of GNU parallel with our launcher
# ... either directly
(access)$> si --ntasks-per-node 4 --exclusive
# ... or using the HPC School reservation 'hpcschool'if needed  - use 'sinfo -T' to check if active and its name
# (access)$> srun --reservation=hpcschool --ntasks-per-node 4 --exclusive --pty bash
```

As before, in another terminal (or another screen tab/windows), connect to that job and run `htop`.
Now you can make some tests:

```bash
# check usage
(node)$> ./launcher.stressme.sh -h   # read [tf] manual, press Q to quit

################ DRY-RUN
(node)$> ./launcher.stressme.sh -n
### Starting timestamp (s): 1607707840
parallel --delay .2 -j 4 --joblog logs/state.parallel.log --resume  srun  --exclusive -n1 -c 1 --cpu-bind=cores /home/users/svarrette/tutorials/sequential/scripts/run_stressme {1} ::: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
### Ending timestamp (s): 1607707840"
# Elapsed time (s): 0

Beware that the GNU parallel option --resume makes it read the log file set by
--joblog (i.e. logs/state*.log) to figure out the last unfinished task (due to the
fact that the slurm job has been stopped due to failure or by hitting a walltime
limit) and continue from there.
In particular, if you need to rerun this GNU Parallel job, be sure to delete the
logfile logs/state*.parallel.log or it will think it has already finished!

############### TEST mode - parallel echo mode (always important to do before running effectively the commands)
(node)$> ./launcher.stressme.sh -t
### Starting timestamp (s): 1607708018
srun --exclusive -n1 -c 1 --cpu-bind=cores /home/users/svarrette/tutorials/sequential/scripts/run_stressme 1
srun --exclusive -n1 -c 1 --cpu-bind=cores /home/users/svarrette/tutorials/sequential/scripts/run_stressme 2
srun --exclusive -n1 -c 1 --cpu-bind=cores /home/users/svarrette/tutorials/sequential/scripts/run_stressme 3
[...]
srun --exclusive -n1 -c 1 --cpu-bind=cores /home/users/svarrette/tutorials/sequential/scripts/run_stressme 27
srun --exclusive -n1 -c 1 --cpu-bind=cores /home/users/svarrette/tutorials/sequential/scripts/run_stressme 28
srun --exclusive -n1 -c 1 --cpu-bind=cores /home/users/svarrette/tutorials/sequential/scripts/run_stressme 29
srun --exclusive -n1 -c 1 --cpu-bind=cores /home/users/svarrette/tutorials/sequential/scripts/run_stressme 30
### Ending timestamp (s): 1607708024"
# Elapsed time (s): 6

Beware that the GNU parallel option --resume makes it read the log file set by
--joblog (i.e. logs/state*.log) to figure out the last unfinished task (due to the
fact that the slurm job has been stopped due to failure or by hitting a walltime
limit) and continue from there.
In particular, if you need to rerun this GNU Parallel job, be sure to delete the
logfile logs/state*.parallel.log or it will think it has already finished!
/!\ WARNING: Test mode - removing sate file

############## REAL run
(node)$> ./launcher.stressme.sh
### Starting timestamp (s): 1607708111
#  /usr/bin/stress -c 1 -t 1
stress: info: [127447] dispatching hogs: 1 cpu, 0 io, 0 vm, 0 hdd
stress: info: [127447] successful run completed in 1s
#  /usr/bin/stress -c 1 -t 2
stress: info: [127439] dispatching hogs: 1 cpu, 0 io, 0 vm, 0 hdd
stress: info: [127439] successful run completed in 2s
[...]
#  /usr/bin/stress -c 1 -t 29
stress: info: [128239] dispatching hogs: 1 cpu, 0 io, 0 vm, 0 hdd
stress: info: [128239] successful run completed in 29s
#  /usr/bin/stress -c 1 -t 30
stress: info: [128276] dispatching hogs: 1 cpu, 0 io, 0 vm, 0 hdd
stress: info: [128276] successful run completed in 30s
### Ending timestamp (s): 1607708243"
# Elapsed time (s): 132

Beware that the GNU parallel option --resume makes it read the log file set by
--joblog (i.e. logs/state*.log) to figure out the last unfinished task (due to the
fact that the slurm job has been stopped due to failure or by hitting a walltime
limit) and continue from there.
In particular, if you need to rerun this GNU Parallel job, be sure to delete the
logfile logs/state*.parallel.log or it will think it has already finished!
```

A quick look in parallel on `htop` report in the second terminal/screen windows demonstrate the usage of only 4 cores as expressed in the slurm job (`--ntasks-per-node 4`):

![](images/screenshot_htop_gnu_parallel_j4_interactive.png)

**IMPORTANT** as highlighted by the ULHPC script: Beware that the GNU parallel option `--resume` makes it read the log file set by `--joblog` (i.e. `logs/state*.log`) to figure out the last unfinished task (due to the fact that the slurm job has been stopped due to failure or by hitting a walltime
limit) and continue from there.
In particular, as we now want to rerun the **same** GNU Parallel job, be sure to delete the
logfile `logs/state*.parallel.log` or your passive job will likely do nothing as it will think it has already finished!

``` bash
(node)$> rm logs/state.parallel.log
```

You can even test on another set of parameters without changing your script:

``` bash
# BEWARE of placing the range within surrounding double quotes!!!
(node)$> ./launcher.stressme.sh -n "{1..10}"   # Dry-run
(node)$> ./launcher.stressme.sh -t "{1..10}"   # Test
(node)$> ./launcher.stressme.sh "{1..10}"      # Real run
(node)$> rm logs/state.parallel.log
```
To avoid cleaning the joblog file, you may want to _exceptionally_ set it to `/dev/null` using the `--joblog` option also supported by our script.

### Passive job submission

Now that you have validated the expected behavior of the launcher script (you may want to test on higher number of tasks per node: GNU parallel will just adapt without any change to the launcher script), it's time to go for a passive run at full capacity.

Exit `htop` in its terminal (press 'q' to exit) and press `CTRL-D` to disconnect to return to the access server.
Quit your interactive job (exit or `CTRL+D`) and submit it as a passive job:

``` bash
# Exit the interactive job
(node)$> exit    # OR CTRL-D
# Note: you may want/need to run it under the dedicated reservation set for the training event
# using sbatch --reservation=[...]
(access)$> sbatch ./launcher.stressme.sh # --joblog /dev/null
```

_Hint_: for the lazy persons, you can still define on the fly the `TASK` variable as follows:

``` bash
(access)$> TASK=$(pwd)/scripts/run_stressme sbatch ./scripts/launcher.parallel.sh
```

A quick look in parallel on `htop` report (`sq` then `sjoin <JOBID>` then `htop`)  in the second terminal/screen windows demonstrate the expected usage optimizing the full node.
As there are "only" 30 parameters tested by default and that the launcher is configured with 28 tasks per node, the time you open `htop` you will likely _not_ see all cores used.
Yet you can repeat the experience for 100 `run_stressme` tasks quite conveniently, **without** changing the launcher script:

``` bash
# Remove the state log
(access)$> rm logs/state.parallel.log
# Note: you may want/need to run it under the dedicated reservation set for the training event
# using sbatch --reservation=[...]
# BEWARE of placing the range within surrounding double quotes!!!
#    --joblog option used **exceptionnally** to disable resume option
(access)$> sbatch ./launcher.stressme.sh --joblog /dev/null "{1..100}"
```

In this case, you can be reassured on your htop usage:

![](images/screenshot_htop_gnu_parallel_j28.png)

_Note_: if you use the default (old) version of `parallel` (GNU parallel 20160222), you may occasionally witness a single core that seems inactive. This is a bug inherent to that version (tied to the controlling process) corrected in more recent version.

Repeat with 10 tasks

``` bash
# Note: you may want/need to run it under the dedicated reservation set for the training event
# using sbatch --reservation=[...]
# BEWARE of placing the range within surrounding double quotes!!!
(access)$> sbatch ./launcher.stressme.sh --joblog /dev/null "{1..10}"
```

And let's collect the aggregated statistics from these 3 jobs (adapt job ID accordingly):

``` bash
(access)$> slist 2175717,2175719,2175721 -X
# sacct -j 2175717,2175719,2175721 --format User,JobID,Jobname%30,partition,state,time,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist,AveCPU,ConsumedEnergyRaw -X
     User        JobID                        JobName  Partition      State  Timelimit    Elapsed     MaxRSS  MaxVMSize   NNodes      NCPUS        NodeList     AveCPU ConsumedEnergyRaw
--------- ------------ ------------------------------ ---------- ---------- ---------- ---------- ---------- ---------- -------- ---------- --------------- ---------- -----------------
svarrette 2175717                         GnuParallel      batch  COMPLETED   01:00:00   00:00:13                              1         28        iris-117                         2430
svarrette 2175719                         GnuParallel      batch  COMPLETED   01:00:00   00:00:36                              1         28        iris-117                         9311
svarrette 2175721                         GnuParallel      batch  COMPLETED   01:00:00   00:03:57                              1         28        iris-117                        70702
[...]
```

Compared to the serial ampersand approach, we have thus **obtained a significant improvement in time efficiency** for the same node occupancy (1  single node allocated):

* for 30 tasks:  **39% improvement** (36s vs 59s), demonstrating a 92% improvement compared to the sequential run.
     - it's also 22% better than the VERY BAD `for [...] sbatch` approach (36s vs 46s)
* for 100 tasks: **13% improvement** (237s vs. 277s), demonstrating a 95% improvement compared to the sequential run.




## Embarrassingly [GNU] parallel tasks across multiples nodes

[GNU Parallel](http://www.gnu.org/software/parallel/) supports the distribution of tasks to multiple compute nodes using ssh connections, _i.e._ via the the `--sshloginfile <filename>` or `--sshlogin` options.

However, **Scaling Parallel with `--sshlogin[file]` is Not Recommended**
Though this allows work to be balance between multiple nodes, past experience suggests that scaling is much less effective. And as the typical usage of GNU parallel includes embarrassingly parallel tasks, there is no real justification to go across multiples nodes with a single job.
For instance, let's assume you wish to explore the parameters `{1..1000}`, you can divide your search space in (for instance) 5 sub domains and dedicated one GnuParallel job (exploiting a full node) for each subdomain as follows:

``` bash
(access)$> sbatch ./launcher.stressme.sh --joblog logs/state.1.parallel.log "{1..200}"
(access)$> sbatch ./launcher.stressme.sh --joblog logs/state.2.parallel.log "{201..400}"
(access)$> sbatch ./launcher.stressme.sh --joblog logs/state.3.parallel.log "{401..600}"
(access)$> sbatch ./launcher.stressme.sh --joblog logs/state.4.parallel.log "{601..800}"
(access)$> sbatch ./launcher.stressme.sh --joblog logs/state.5.parallel.log "{801..1000}"
```

Each of these jobs will cover (independently) the subdomain.
Note that you **MUST** set different joblog files to avoid collusion -- you can use the `--joblog` option supported by our launcher.

Finally, if you would need to scale to more than 5 such jobs, you are encouraged to use the [job dependency mechanism](https://slurm.schedmd.com/sbatch.html) implemented by Slurm to limit the number of concurrent running nodes.
This can be easily achieved with the `singleton` dependency (option `-d` of sbatch)  and carefully selected job names (option `-J` of sbatch):

> `-d, --dependency=singleton`: This job can begin execution **after any previously launched jobs sharing the same job name and user have terminated**. In other words, only one job by that name and owned by that user can be running or suspended at any point in time.

This would permit to restrict the number of concurrent jobs (leaving more resources for others)
For example, to explore `{1..2000}` (and more generally `{$min..$max}`), without exceeding concurrent running on more than 4 nodes (i.e. within a "tunnel" of 4 nodes max), you could proceed as follows:

``` bash
# Abstract search space parameters
min=1
max=2000
chunksize=200
for i in $(seq $min $chunksize $max); do
    ${CMD_PREFIX} sbatch \
                  -J ${JOBNAME}_$(($i/$chunksize%${MAXNODES})) --dependency singleton \
                  ${LAUNCHER} --joblog log/state.${i}.parallel.log  "{$i..$((i+$chunksize))}";
done
```

A sample submission script [`scripts/submit_stressme_multinode`](scripts/submit_stressme_multinode) is proposed to illustrate this concept:

```bash
(access)$> ./scripts/submit_stressme_multinode -h
Usage: submit_stressme_multinode [-x] [-N MAXNODES]
    Sample submision script across multiple nodes
    Execution won t spread on more than 4 nodes (singleton dependency)
      -x --execute         really submit the jobs with sbatch
      -N --nodes MAXNODES  set max. nodes

# Target restriction to 4 running nodes max
(access)$> ./scripts/submit_stressme_multinode
sbatch -J StressMe_0 --dependency singleton launcher.stressme.sh --joblog logs/state.1.parallel.log {1..201}
sbatch -J StressMe_1 --dependency singleton launcher.stressme.sh --joblog logs/state.201.parallel.log {201..401}
sbatch -J StressMe_2 --dependency singleton launcher.stressme.sh --joblog logs/state.401.parallel.log {401..601}
sbatch -J StressMe_3 --dependency singleton launcher.stressme.sh --joblog logs/state.601.parallel.log {601..801}
sbatch -J StressMe_0 --dependency singleton launcher.stressme.sh --joblog logs/state.801.parallel.log {801..1001}


# Target restriction to 2 running nodes max
(access)$> ./scripts/submit_stressme_multinode -N 2
sbatch -J StressMe_0 --dependency singleton launcher.stressme.sh --joblog logs/state.1.parallel.log {1..201}
sbatch -J StressMe_1 --dependency singleton launcher.stressme.sh --joblog logs/state.201.parallel.log {201..401}
sbatch -J StressMe_0 --dependency singleton launcher.stressme.sh --joblog logs/state.401.parallel.log {401..601}
sbatch -J StressMe_1 --dependency singleton launcher.stressme.sh --joblog logs/state.601.parallel.log {601..801}
sbatch -J StressMe_0 --dependency singleton launcher.stressme.sh --joblog logs/state.801.parallel.log {801..1001}
```

Now if you execute it, you will see only two running jobs and thus nodes (the other jobs are waiting for the dependency to be met)

```bash
(access)$> ./scripts/submit_stressme_multinode -N 2 -x
Submitted batch job 2175778
Submitted batch job 2175779
Submitted batch job 2175780
Submitted batch job 2175781
Submitted batch job 2175782

(access)$> sq
# squeue -u svarrette
   JOBID PARTIT    QOS        NAME  NODE  CPUS ST  TIME TIME_LEFT NODELIST(REASON)
 2175780  batch normal  StressMe_0     1    28 PD  0:00   1:00:00 (Dependency)
 2175781  batch normal  StressMe_1     1    28 PD  0:00   1:00:00 (Dependency)
 2175782  batch normal  StressMe_0     1    28 PD  0:00   1:00:00 (Dependency)
 2175779  batch normal  StressMe_1     1    28  R  0:02     59:58 iris-064
 2175778  batch normal  StressMe_0     1    28  R  0:05     59:55 iris-047
```
