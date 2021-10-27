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
cluster = SLURMCluster(cores=os.environ.get("SLURM_CPUS_PER_TASK",1),
                       processes=1,
                       memory="4GB",
                       walltime="01:00:00",
                       queue="batch",
                       interface="ib0")

numworkers = os.environ("NB_WORKERS",1)
cluster.scale(numworkers)

# Connect to distributed cluster and override default
client = Client(cluster)

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

# Second approach to delayed function
total = dask.delayed(sum)(output)
total.visualize(filename='task_graph.svg')
# parallel execution workers
results = total.compute()
print(results)
#### Very important ############
cluster.close()

