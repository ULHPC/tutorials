-*- mode: markdown;mode:visual-line;  fill-column: 80 -*-

Authors: Clément Parisot

Copyright (c) 2018 UL HPC Team  -- see <http://hpc.uni.lu>

---------------------------------------------------------
# UL HPC Tutorial: [Advanced] Python : Use Jupyter notebook on UL HPC

[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/python/advanced/tutorial_python_advanced.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/python/advanced) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/python/advanced/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

[![](https://github.com/ULHPC/tutorials/raw/devel/python/advanced/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/python/advanced/tutorial_python_advanced.pdf)

Python is a high-level interpreted language widely used in research. It lets you work quickly and comes with a lot of available packages which give more useful functionalities.

# Use Celery on Iris

## Choose a broker

### Redis broker

We need to run our own instance of Redis server on UL HPC on a node. We will download the executable from redis.io website and execute it locally on a node.

Reserve a node interactively and do the following:

```bash
wget http://download.redis.io/releases/redis-4.0.9.tar.gz
tar xzf redis-4.0.9.tar.gz
cd redis-4.0.9
make
```

Let's create a configuration file for redis-server with the following options:

* port where the server is listening (default one): 6379
* ip address of the server: we will listen on the main ethernet interface of the node. You can retrieve the IP address with this command `facter ipaddress` or by checking the output of `ip addr show dev em1`.
* we will protect the access to the node with a password to ensure that other experiments doesn't interact with us.

Which gives us the following config file:

```bash
cat > redis2.conf << END
protected-mode yes
port 6379
requirepass yourverysecurepassword
END
```

Now you should be able to run the Redis server on a node with this configuration.

The resources are by default shared with other users. You can't run a redis instance on the same resource (same IP) with the same port number. To avoid collision with other users, you should either reserve a full node to be sure to be the only one running a Redis instance with this IP or if you want to share the IP of your node with somebody else, make sure to use a different port number.

We will run our redis server on a different port number for each run by using this bash command: `$(($SLURM_JOB_ID % 1000 + 64000))`. It will give us a port number between 64000 and 64999 based on the last 3 digits of our job ID.

```bash
si -J redis-server
./src/redis-server $HOME/celery/redis/redis.conf --port $(($SLURM_JOB_ID % 1000 + 64000)) --bind $(facter ipaddress)
```

You should have the following output:
```
7625:C 24 May 14:06:06.280 # oO0OoO0OoO0Oo Redis is starting oO0OoO0OoO0Oo
7625:C 24 May 14:06:06.280 # Redis version=4.0.9, bits=64, commit=00000000, modified=0, pid=7625, just started
7625:C 24 May 14:06:06.280 # Configuration loaded
7625:M 24 May 14:06:06.281 * Increased maximum number of open files to 10032 (it was originally set to 1024).
                _._
           _.-``__ ''-._
      _.-``    `.  `_.  ''-._           Redis 4.0.9 (00000000/0) 64 bit
  .-`` .-```.  ```\/    _.,_ ''-._
 (    '      ,       .-`  | `,    )     Running in standalone mode
 |`-._`-...-` __...-.``-._|'` _.-'|     Port: 64856
 |    `-._   `._    /     _.-'    |     PID: 7625
  `-._    `-._  `-./  _.-'    _.-'
 |`-._`-._    `-.__.-'    _.-'_.-'|
 |    `-._`-._        _.-'_.-'    |           http://redis.io
  `-._    `-._`-.__.-'_.-'    _.-'
 |`-._`-._    `-.__.-'    _.-'_.-'|
 |    `-._`-._        _.-'_.-'    |
  `-._    `-._`-.__.-'_.-'    _.-'
      `-._    `-.__.-'    _.-'
          `-._        _.-'
              `-.__.-'

7625:M 24 May 14:06:06.283 # WARNING: The TCP backlog setting of 511 cannot be enforced because /proc/sys/net/core/somaxconn is set to the lower value of 128.
7625:M 24 May 14:06:06.283 # Server initialized
7625:M 24 May 14:06:06.283 # WARNING overcommit_memory is set to 0! Background save may fail under low memory condition. To fix this issue add 'vm.overcommit_memory = 1' to /etc/sysctl.conf and then reboot or run the command 'sysctl vm.overcommit_memory=1' for this to take effect.
7625:M 24 May 14:06:06.283 # WARNING you have Transparent Huge Pages (THP) support enabled in your kernel. This will create latency and memory usage issues with Redis. To fix this issue run the command 'echo never > /sys/kernel/mm/transparent_hugepage/enabled' as root, and add it to your /etc/rc.local in order to retain the setting after a reboot. Redis must be restarted after THP is disabled.
7625:M 24 May 14:06:06.283 * DB loaded from disk: 0.000 seconds
7625:M 24 May 14:06:06.283 * Ready to accept connections
```

Now you should be able to connect to your redis server from the other nodes and from the access. You can test it simply with telnet from access.iris. Open a new connection to iris-cluster and type the following command:

```bash
telnet iris-001 64856 # Please replace iris-001 by your node name and 64856 by your port number
AUTH myverysecurepassword
PING
+PONG
^]
telnet> quit
Connection closed.
```

To exit `telnet` strike **Ctrl-]** keys.

## Celery configuration

All information comes from the [official documentation of celery](http://docs.celeryproject.org/en/latest/getting-started/brokers/redis.html#broker-redis)

### Installation

* Reserve a node interactively on iris
* Create a virtual environment
* Install `celery[redis]` inside the virtualenv using pip installer

```bash
si
cd celery
module load lang/Python/3.7  # actual version depends on the selected software set
virtualenv venv
source venv/bin/activate
pip install "celery[redis]"
pip install redis
# (optional) Flower is a frontend for visualization of the queues status
pip install flower
```

### Configuration

We need to give to celery 3 informations about our Redis:
* **password** of the database
* **hostname** of the node on which the server is running
* **port** the port number of the database

As those parameters will change on each run, we will put the 3 value inside a configuration file and import it in the python code to create the broker address which will looks like this:

```python
redis_broker = "redis://:{password}@{hostname}:{port}/0".format(**params)
```

In file `celery.ini`, fill the redis section like this:

```
[redis]
broker_password=<ỳourverysecurepassword>
broker_hostname=<hostname of the redis server that can be find with `squeue -u $USER`>
broker_port=<port that you have defined in the configuration (by default 64867)
```

We have created a list of tasks to execute in `ulhpccelery/tasks.py`. There are 3 tasks:

* add(x, y) add x and y number
* mul(x, y) multiplie x and y
* xsum(numbers) return the sum of an array of numbers

We will start a worker on a full node that will run the code on the 28 cores of iris. For that, reserve a full node and 28 cores, load the virtual environment and run celery.

```
si -N1 -c28
cd celery
module load lang/Python/3.6.0
source venv/bin/activate
celery -A ulhpccelery worker
```

You should see the working starting on the 28 cores and connect to the redis instance successfully. If you have issue connecting to the redis instance, check that it is still running and that you have access to it from the node (via telnet command for example).

### Launch several tasks

From the **ulhpccelery** module, simply reserve a node and execute the following commands.

```
si -N1 -c1
cd celery
module load lang/Python/3.6.0
source venv/bin/activate
python
>> from ulhpccelery import tasks
>> res = []
>> for i in range(10**6):
>>     res.append(tasks.add.delay(i, i+1))
>> for r in res:
>>     print(r.get())
```

You should see the results of the additions. The tasks have been distributed to all the available cores.

### Monitor the experiment

You can use Flower to monitor the usage of the queues.

```
si -N1 -c28
cd celery
module load lang/Python/3.6.0
virtualenv venv
source venv/bin/activate
celery -A ulhpccelery flower --address="$(facter ipaddress)"
```

Now, directly access to the web interface of the node (after a tunnel redirection): http://172.17.6.55:5555/

You should see this kind of output:

![Flower interface](./celery/Flower.png)

### To go further

* Try to add / suppress workers during the execution
* Try to stop restart redis server
