---
fontsize: 12pt
---

[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/python/advanced/scikit-learn/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/python/advanced/scikit-learn/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/python/advanced/scikit-learn/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)


# Scalable computing with Dask

![](./images/dask_logo.jpeg)

## Description

Dask is a flexible library to perform parallel computing Data Science tasks in [Python](https://www.python.org/). Although multiple parallel and distributed computing libraries already exist in Python, Dask remains **Pythonic** while being very efficient (see [Diagnosing Performance](https://distributed.dask.org/en/latest/diagnosing-performance.html)).

Dask is composed of two parts:

* **Dynamic task scheduling**: Optimized computational workloads (see [distributed dask](https://distributed.dask.org/en/latest/))
* **Big Data collections**: Parallel and distributed equivalent data collecting extending [Numpy](https://numpy.org/) array, [Pandas](https://pandas.pydata.org/) dataframes

An interesting feature of Dask is Python iterators for large-than-memory or distributed environments. Dask tries to provide different qualities:

* **Familiar**: Provides parallelized NumPy array and Pandas DataFrame objects

* **Flexible**: Provides a task scheduling interface for more custom workloads and integration with other projects.

* **Native**: Enables distributed computing in pure Python with access to the PyData stack.

* **Fast**: Operates with low overhead, low latency, and minimal serialization necessary for fast numerical algorithms

* **Scales up**: Runs resiliently on clusters with 1000s of cores

* **Scales down**: Trivial to set up and run on a laptop in a single process

* **Responsive**: Designed with interactive computing in mind, it provides rapid feedback and diagnostics to aid humans


![](https://docs.dask.org/en/latest/_images/dask-overview.svg)


## Task graphs

Dask solely rely on a graphs representation to encode algorithms. The main advantage of these structures is a clear and efficient approach for task scheduling. For Dask users, task graphs and operations are fully transparent unless you decide to develop a new module. 

<style>
.tab_css {
  width: 100%;
}
.tab_css td,th{
   border: none!important;
}
</style>

<table class="tab_css">
<tr>
<th> Code </th>
<th> Task graph </th>
</tr>
<tr>
<td>
```python
def inc(i):
    return i + 1

def add(a, b):
    return a + b

x = 1
y = inc(x)
z = add(y, 10)

# The graph is encoded as a dictionnary
d = {'x': 1,
     'y': (inc, 'x'),
     'z': (add, 'y', 10)}
```
</td>
<td>
<img style="float: right;max-height: 200px;" src="https://docs.dask.org/en/latest/_images/dask-simple.png">
</td>
</tr>
</table>


## Install Dask

Dask can be installed with pip or conda like any other python package.

### Anaconda/Conda

```bash
conda install dask
# OR
conda install dask-core # Use it if you need a minimal version of dask
```
Please note that Dask is already included by default in the [Anaconda distribution](https://www.anaconda.com/download).

### Pip

In order to install all dependencies (e.g. NumPy, Pandas, ...), use the following command:

```bash
python -m pip install "dask[complete]"
# OR simply
pip install "dask[complete]" 
```
Similarly to conda, dask core can be install with the command `pip install dask`. Note that additionnal modules like dask.array, dask.dataframe could be separately installed. However we strongly recommend to proceed with a full installation.

```bash
pip install "dask[array]"       # Install requirements for dask array
pip install "dask[dataframe]"   # Install requirements for dask dataframe
pip install "dask[diagnostics]" # Install requirements for dask diagnostics
pip install "dask[distributed]" # Install requirements for distributed dask
```

### Install from sources

For those wishing to compile and optimize the library on a dedicated hardware, Dask can be compiled and installed as follows:

```bash
git clone https://github.com/dask/dask.git
cd dask
python -m pip install .
# OR
pip install ".[complete]" 
```

### On the ULHPC platform

We strongly recommend to install Dask inside a virtual environment using the python versions included in the [software set](https://hpc-docs.uni.lu/software/). 

```bash
# See https://hpc-docs.uni.lu/connect/ssh/
ssh [aion,iris]-cluster    # assuming proper configuration
# Once on the clusters, ask for a interactive job
si --time=01:00:00 # OR si-gpu --time=01:00:00 if a GPU is needed
module load lang/Python # Load default python 
python -m venv dask_env_${ULHPC_CLUSTER}
source dask_env_${ULHPC_CLUSTER}/bin/activate
pip install --upgrade pip
pip install "dask[complete]"
```

## Setup


Dask can be used on different hardware going from your laptop to a multi-node cluster. For this purpose, Dask considers two families of task schedulers.
By default, if no client is instantiated, Dask will turn on the local schedule. 


```python
import dask.dataframe as dd
df = dd.read_csv(...)
df.x.sum().compute()  # This uses the single-machine scheduler by default
```

If you need more resources, [dask.distributed](https://distributed.dask.org/en/latest/) will be needed to setup and connect to a distributed cluster.

```python
from dask.distributed import Client
client = Client(...)  # Connect to distributed cluster and override default
df.x.sum().compute()  # This now runs on the distributed system
```

## Setting a Dask cluster

In the remainder of this paper, we will only consider Distributed Dask cluster. Nevertheless, you can also consider a local cluster on your laptop to test your workflow at small scale. More details can be found in the [dask.distributed documentation](https://distributed.dask.org/en/latest/).


On the ULHPC platform, you have two strategies to create a Dask cluster:

- Using [`SLURMCluster` class](https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SLURMCluster.html)
- Starting manually the dask-scheduler and dask-workers 


First, we are going to setup a python virtual environment in order to install all required python libraries. 

Be sure to start with a bare environment:

* No interactive job running and thus no loaded modules
* No python virtualenv already loaded

Apply the following commands to setup your environment.

```bash
# Clone tutorial repository
git clone https://github.com/ULHPC/tutorials.git
# cd into the scripts folder
cd tutorials/python/advanced/dask-ml/scripts
# Ask an interactive job
si --time=01:00:00
# Load python3 module (load by default Python3)
module load lang/Python
python -m venv dask_env_${ULHPC_CLUSTER}
source dask_env_${ULHPC_CLUSTER}/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```




### Automatic setup 

We first create a generic launcher `cluster_jobs_workers.sh`. Be carefull, you will need to install two different virtualenv if you planned to run the code on both clusters, i.e., Aion and Iris.

```bash
#!/bin/bash -l

#SBATCH -p batch          
#SBATCH -J DASK_jobs_workers     
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                
#SBATCH --cpus-per-task=1               
#SBATCH -t 00:30:00        

# Load the python version used to install Dask
module load lang/Python

# Make sure that you have an virtualenv dask_env installed
export DASK_VENV="$1" 
shift
if [ ! -d "${DASK_VENV}" ] || [ ! -f "${DASK_VENV}/bin/activate" ]; then
    
    echo "Error with virtualenv" && exit 1

fi

# Source the python env
source "${DASK_VENV}/bin/activate"

python -u $*
```

Below a small example to start workers as slurm jobs: `cluster_jobs_workers.py`.

```python
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import dask
# Library to generate plots
import matplotlib as mpl
# Define Agg as Backend for matplotlib when no X server is running
mpl.use('Agg')
import matplotlib.pyplot as plt
import socket
import os

# Submit workers as slurm job
# Below we define the slurm parameters of a single worker
cluster = SLURMCluster(cores=os.envion.get("SLURM_CPUS_PER_TASK",1),
                       processes=1,
                       memory="4GB",
                       walltime="01:00:00",
                       queue="batch",
                       interface="ib0")

numworkers = os.environ("SLURM_NTASKS",1)
cluster.scale(numworkers)

# Connect to distributed cluster and override default
client = Client(cluster)
client.wait_for_workers()

# Decorator  
@dask.delayed
def inc(x):
    return x + 1

@dask.delayed
def double(x):
    return x * 2

@dask.delayed
def add(x, y):
    return x + y

data = [1, 2, 3, 4, 5]

output = []
for x in data:
    a = inc(x)
    b = double(x)
    c = add(a, b)
    output.append(c)

# Second approach as a delayed function
total = dask.delayed(sum)(output)
total.visualize(filename='task_graph.svg')
# parallel execution workers
results = total.compute()
print(results)
#### Very important ############
cluster.close()
```
The Dask delayed function decorates your functions so that they operate lazily. Rather than executing your function immediately, it will defer execution, placing the function and its arguments into a task graph.

![](https://docs.dask.org/en/latest/_images/delayed-inc-double-add.svg)

You can execute the previous example with the following command: `sbatch cluster_jobs_workers.sh dask_env_${ULHPC_CLUSTER} cluster_jobs_workers.py`. Once the main job has started, you should see dask-workers spanning in the queue using `squeue -u user`.
Please note also that each worker has his own slurm-**jobid**.out file which provide all necessary information to diagnose problems. An example is provided below.

```bash
distributed.nanny - INFO -         Start Nanny at: 'tcp://172.19.6.19:44324'
distributed.worker - INFO -       Start worker at:    tcp://172.19.6.19:37538
distributed.worker - INFO -          Listening to:    tcp://172.19.6.19:37538
distributed.worker - INFO -          dashboard at:          172.19.6.19:39535
distributed.worker - INFO - Waiting to connect to:    tcp://172.19.6.19:39227
distributed.worker - INFO - -------------------------------------------------
distributed.worker - INFO -               Threads:                          1
distributed.worker - INFO -                Memory:                   3.73 GiB
distributed.worker - INFO -       Local Directory: /mnt/irisgpfs/users/ekieffer/Dask/dask-worker-space/worker-p1ij_9ar
distributed.worker - INFO - -------------------------------------------------
distributed.worker - INFO -         Registered to:    tcp://172.19.6.19:39227
distributed.worker - INFO - -------------------------------------------------
distributed.core - INFO - Starting established connection
distributed.worker - INFO - Stopping worker at tcp://172.19.6.19:37538
distributed.nanny - INFO - Worker closed
distributed.nanny - INFO - Closing Nanny at 'tcp://172.19.6.19:44324'
distributed.dask_worker - INFO - End worker
```

### Manual setup

In this part, we show how to start manually the dask-scheduler and how to spawn the workers. Hereafter, workers will not be created as new jobs but new steps inside a main job. We advise to use this workflow to avoid filling the scheduler queue.

First, we create a new slurm launcher: `cluster_steps_workers.sh`

```bash
#!/bin/bash -l

#SBATCH -p batch    
#SBATCH -J DASK_steps_workers    
#SBATCH -N 2
#SBATCH -n 10     
#SBATCH -c 1    
#SBATCH -t 00:30:00    

# Load the python version used to install Dask
module load lang/Python

# Make sure that you have an virtualenv dask_env installed
export DASK_VENV="$1" 
shift
if [ ! -d "${DASK_VENV}" ] || [ ! -f "${DASK_VENV}/bin/activate" ]; then
        
        echo "Error with virtualenv" && exit 1

    fi

# Source the python env 
source "${DASK_VENV}/bin/activate"

# Dask configuration to store the scheduler file
DASK_CONFIG="${HOME}/.dask"
DASK_JOB_CONFIG="${DASK_CONFIG}/job_${SLURM_JOB_ID}"
mkdir -p ${DASK_JOB_CONFIG}
export SCHEDULER_FILE="${DASK_JOB_CONFIG}/scheduler.json"


# Number of tasks - 1 controller task - 1 python task
export NB_WORKERS=$((${SLURM_NTASKS}-2))


LOG_DIR="$(pwd)/logs/job_${SLURM_JOBID}"
mkdir -p ${LOG_DIR}

# Start controller on this first task
srun -w $(hostname) --output=${LOG_DIR}/ipcontroller-%j-workers.out  --exclusive -N 1 -n 1 -c ${SLURM_CPUS_PER_TASK} \
     dask-scheduler  --scheduler-file "${SCHEDULER_FILE}"  --interface "ib0" &
sleep 10

#srun: runs ipengine on each other available core
srun --output=${LOG_DIR}/ipengine-%j-workers.out \
     --exclusive -n ${NB_WORKERS} -c ${SLURM_CPUS_PER_TASK} \
     --cpu-bind=cores dask-worker  \
     --label \
     --interface "ib0" \
     --scheduler-file "${SCHEDULER_FILE}"  &

sleep 25 

srun --output=${LOG_DIR}/code-%j-execution.out  --exclusive -N 1 -n 1 -c ${SLURM_CPUS_PER_TASK} python -u $*

```

** Remark ** The launcher below requests 10 tasks on 2 nodes with 1 cpu per task. This is **NOT** an efficient use of the hardware but only for educational purpose. Please always try to maximize nodes usage, i.e., 28 tasks max on iris, 128 max on aion or decrease and increase multithreading if possible. You may use  `--ntasks-per-nodes`or `--ntasks-per-socket` for this purpose. Please also refer to the [ULHPC documentation](https://hpc-docs.uni.lu/slurm/#specific-resource-allocation) for more details. 


To illustrate this manual setting, we are going now to scale XGBoost using Dask. [XGBoost](https://xgboost.readthedocs.io/en/latest/) is an optimized gradient boosting library designed to be highly efficient, flexible and portable. Gradient boosted trees can be distributed by making Dask and XGBoost working together. XGBoost provides a powerful prediction framework, and it works well in practice. It wins Kaggle contests and is popular in industry because it has good performance and can be easily interpreted (i.e., it’s easy to find the important features from a XGBoost model).

<table class="tab_css">
<tr>
</tr>
<tr>
<td>
<img style="float: center;height: 100px;" src="https://dask.readthedocs.io/en/latest/_images/dask_horizontal.svg">
</td>
<td>
<img style="float: center;height: 100px;" src="https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/xgboost.png">
</td>
</tr>
</table>


Suppose we have access to Dask cluster with a set of workers. The first task is to install the xgboost library.

```python
pip install xgboost
```

then create the following script `cluster_steps_workers.py`:

```python
from dask.distributed import Client
# Library to generate plots
import matplotlib as mpl
# Define Agg as Backend for matplotlib when no X server is running
mpl.use('Agg')
import matplotlib.pyplot as plt
import dask
import xgboost as xgb
import dask.array as da
import json
import os

data=[]
# Using the distributed shared file system, we can access to the Dask cluster
# configuration.
# We read the scheduler address and port from the scheduler file
with open(os.environ["SCHEDULER_FILE"]) as f:
        data = json.load(f)
        scheduler_address=data['address']

# Connect to the the cluster
client = Client(scheduler_address)
client.wait_for_workers()

# X and y must be Dask dataframes or arrays
num_obs = 1e5
num_features = 20
X = da.random.random(size=(num_obs, num_features), chunks=(1000, num_features))
y = da.random.random(size=(num_obs, 1), chunks=(1000, 1))

# Training
dtrain = xgb.dask.DaskDMatrix(client, X, y)

output = xgb.dask.train(
    client,
    {"verbosity": 2, "tree_method": "hist", "objective": "reg:squarederror"},
    dtrain,
    num_boost_round=10,
    evals=[(dtrain, "train")],
)

booster = output['booster']  # booster is the trained model
history = output['history']  # A dictionary containing evaluation results

ax = xgb.plot_importance(booster, height=0.8, max_num_features=9)
ax.grid(False, axis="y")
ax.set_title('Estimated feature importance')
plt.savefig("importance.png")

# Stop Dask cluster
client.shutdown()
```

You can try the manual setup example using the following command: `sbatch cluster_steps_workers.sh dask_env_${ULHPC_CLUSTER} cluster_steps_workers.py`.


## Remark on the worker Thread Pool

Each worker sends computations to a thread in a concurrent.futures.ThreadPoolExecutor for computation. These computations occur in the same process as the Worker communication server so that they can access and share data efficiently between each other. For the purposes of data locality all threads within a worker are considered the same worker.

If your computations are mostly numeric in nature (for example NumPy and Pandas computations) and release the GIL entirely then it is advisable to run dask-worker processes with many threads and one process. This reduces communication costs and generally simplifies deployment.

If your computations are mostly Python code and don’t release the GIL then it is advisable to run dask-worker processes with many processes and one thread per process:

```bash
$ dask-worker scheduler:8786 --nprocs 8 --nthreads 1
```

This will launch 8 worker processes each of which has its own ThreadPoolExecutor of size 1.

If your computations are external to Python and long-running and don’t release the GIL then beware that while the computation is running the worker process will not be able to communicate to other workers or to the scheduler. This situation should be avoided. If you don’t link in your own custom C/Fortran code then this topic probably doesn’t apply.


# References

* https://distributed.dask.org/en/latest/
* https://jobqueue.dask.org/en/latest/
* https://wiki.mpimet.mpg.de/doku.php?id=analysis:pot_pourri:sapphire:dask_parallel_postprocessing

