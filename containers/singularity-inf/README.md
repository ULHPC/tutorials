[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/advanced/advanced_scheduling/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/advanced/advanced_scheduling/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# UL HPC Tutorial: Singularity with Infiniband

     Copyright (c) 2018-2021 UL HPC Team <hpc-team@uni.lu>




<p align="center">
<img src="./images/sing.png" width="300px" >
</p>

## Singularity setups

Please refer to the [Singularity introduction tutorial](/containers/singularity/) for the setups.

## Singularity with Infiniband

* Objectives: (through this example we get the singularity configuration for running over Infiniband, tested on Iris cluster)
  - Create a container with the required Infiniband libraries on Ubuntu18.04
  - Install KerA (stream storage, similar to Apache Kafka)
  - Install DFI (data flow interface over RDMA see https://doi.org/10.1145/3448016.3452816)
  - Run a distributed KerA (1 coordinator and 2 brokers)
  - Configuration validated on the Iris cluster

### Step 1: Required software

* Install a VM, user zetta, with Ubuntu 18.04 having docker and singularity installed.
* Git clone KerA from https://gitlab.uni.lu/ocm/kera.git and follow Step 1 from https://gitlab.uni.lu/ocm/kera.git (git submodule update --init --recursive)
* The kera project contains a Dockerfile and its docker-entry-point.sh

### Step 2: Create the docker container

* Clean and create the kera docker that will be later used by singularity

```bash
sudo docker system prune -a
#Update kera/GNUmakefile JAVAHOME variable, set to JAVAHOME := /root/.sdkman/candidates/java/current
sudo docker build . --tag kera
```

* The Dockerfile contains steps to install required libraries for compiling KerA and DFI, including the right Mellanox driver for Iris

```bash
# ##############################################################################
# 
# Builder stage
#
# ##############################################################################


FROM ubuntu:18.04 AS builder

ARG JAVA_VERSION=11.0.2-open
ENV SDKMAN_DIR=/root/.sdkman

WORKDIR /opt
COPY . /opt/kera

#
# Install dependencies
#

RUN apt-get update \ 
 &&  DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

RUN apt-get update \
 && apt-get install --yes \
      apt-transport-https \
      build-essential \
      apt-utils \
      ca-certificates \
      curl \
      doxygen \
      g++ \
      gdb \
      git \
      libboost-filesystem-dev \
      libboost-program-options-dev \
      libboost-system-dev \
      libibverbs-dev \
      libpcre++-dev \
      libssl-dev \
      libzookeeper-mt-dev \
      procps \
      protobuf-compiler \
      python3 \
      python3-pip \
      software-properties-common \
      unzip \
      wget \
      zip \
      gcc \
      make \
      perl \
      dkms \
      linux-headers-$(uname -r) \
      gnupg \
      lsb-release \
      libprotobuf-dev \
      libcrypto++-dev \
      libevent-dev \
      libboost-all-dev \
      libpcre3-dev \
      libgtest-dev \
      zookeeper \
      tk \
      libnl-3-dev \
      udev \
      tcl \
      libnl-route-3-dev \
      bison \
      flex \
      libmnl0 \
      gfortran \
      libgfortran4 \
      cmake libtool pkg-config autoconf automake libzmq3-dev libgtest-dev libnuma-dev libcppunit-dev numactl libaio-dev libevent-dev \
 && rm -rf /var/lib/apt/lists/*

RUN cd /usr/src/gtest && cmake CMakeLists.txt && make && cp *.a /usr/lib

RUN cd kera \
 && mnt/MLNX_OFED_LINUX-5.1-2.5.8.0-ubuntu18.04-x86_64/mlnxofedinstall  --skip-unsupported-devices-check --without-dkms --add-kernel-support --kernel 5.4.0-42-generic --kernel-sources /usr/src/linux-headers-5.4.0-42-generic/ --without-fw-update --force

RUN cd kera/DFI \
 && mkdir release \
 && cd release/ \
 && cmake .. -DCMAKE_BUILD_TYPE=Release \
 && make \
 && make install


#
# Install Java
#

RUN curl -s "https://get.sdkman.io" | bash \
 && echo "sdkman_auto_answer=true" > $SDKMAN_DIR/etc/config \
 && echo "sdkman_auto_selfupdate=false" >> $SDKMAN_DIR/etc/config

# Source sdkman to make the sdk command available and install java candidate
RUN bash -c "source $SDKMAN_DIR/bin/sdkman-init.sh && sdk install java $JAVA_VERSION"

# Add candidate path to $PATH environment variable
ENV JAVA_HOME="$SDKMAN_DIR/candidates/java/current"
ENV PATH="$JAVA_HOME/bin:$PATH"

#
# Install cmake
#

RUN curl -sSL https://cmake.org/files/v3.11/cmake-3.11.1-Linux-x86_64.tar.gz | tar -xzC . \
 && mv cmake-3.11.1-Linux-x86_64 cmake

ENV PATH="/opt/cmake/bin/:$PATH"

#
# Compile KerArrow
#

RUN cd /opt/kera/kerarrow \
 && mkdir -p cpp/release \
 && cd cpp/release/ \
 && cmake .. -DCMAKE_BUILD_TYPE=Release -DARROW_PLASMA=on \
 && make -j12 \
 && make install 


#
# Compile KerA
#


RUN cd kera \
 && make clean \
 && make -j12 DEBUG=no INFINIBAND=yes \
 && make install


# ##############################################################################
# 
# Final stage
# No JDK is included in the final image.
#
# ##############################################################################

FROM ubuntu:18.04

#
# Installing binaries
#

COPY --from=builder \
 /opt/kera/install/bin /opt/kera/install/bin

COPY --from=builder \
 /etc/init.d/* /etc/init.d/

COPY --from=builder \
 /usr/local/bin/plasma_store \
 /opt/kera/install/bin/

#
# Installing libraries
#

COPY --from=builder \
 /opt/kera/install/lib/kera/* \
 /opt/kera/install/lib/kera/

COPY --from=builder \
 /etc/* /etc/

COPY --from=builder \
 /usr/lib/x86_64-linux-gnu/libibverbs/libmlx5-rdmav25.so /usr/lib/x86_64-linux-gnu/libibverbs/

COPY --from=builder \
 /opt/kera/install/lib/kera/lib* \
 /opt/kera/kerarrow/cpp/release/release/lib* \
 /usr/local/lib/

COPY --from=builder \
 /usr/lib/x86_64-linux-gnu/libzookeeper* \
 /usr/lib/x86_64-linux-gnu/libboost* \
 /usr/lib/x86_64-linux-gnu/libprotobuf* \
 /usr/lib/x86_64-linux-gnu/libpcre* \
 /usr/lib/x86_64-linux-gnu/libibverbs* \
 /usr/lib/x86_64-linux-gnu/libssl* \
 /usr/lib/x86_64-linux-gnu/libcrypto* \
 /usr/lib/x86_64-linux-gnu/lib* \
 /usr/lib/x86_64-linux-gnu/

COPY --from=builder \
 /lib/x86_64-linux-gnu/* /lib/x86_64-linux-gnu/

COPY --from=builder \
 /usr/include/infiniband/* /usr/include/infiniband/

COPY --from=builder \
 /usr/include/* /usr/include/

COPY --from=builder \
 /lib/modules/* /lib/modules/

COPY --from=builder \
 /usr/local/lib/lib* /usr/local/lib/

COPY --from=builder \
 /usr/local/include/dfi /usr/local/include/dfi

ENV LD_LIBRARY_PATH="/usr/local/lib/:$LD_LIBRARY_PATH"

RUN ldd /usr/local/lib/libkera.so \
 && ldd /opt/kera/install/bin/coordinator \
 && ldd /opt/kera/install/bin/server \
 && ldd /usr/local/lib/libdfi.so

COPY ./docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
```

* The docker-entrypoint.sh will later be used when running the singularity container on the Iris cluster.

```bash
#!/bin/bash

set -e


case "$1" in
    sh|bash)
        set -- "$@"
        exec "$@"
    ;;
    coordinator)
        shift
        echo "Running coordinator on `hostname` ${SLURM_PROCID}"
        /opt/kera/install/bin/coordinator "$@" 1>$HOME/coordinator.out 2>&1 &
        status=$?
        if [ $status -ne 0 ]; then
            echo "Failed to start KerA coordinator: $status"
            exit $status
        fi

        exec tail -f $(ls -Art $HOME/coordinator.out | tail -n 1)
    ;;
    broker)
        if [[ ${SLURM_PROCID} -eq 0 ]]; then
         echo "Doing nothing `hostname` $HOSTNAME"
        else
         echo "Installing on `hostname` $HOSTNAME"
        # The plasma store creates the /tmp/plasma socket on startup
        /opt/kera/install/bin/plasma_store -m 1000000000 -s /tmp/plasma 1>$HOME/plasma-${SLURM_PROCID}.out 2>&1 &
        status=$?
        if [ $status -ne 0 ]; then
            echo "Failed to start KerA server: $status"
            exit $status
        fi

        sleep 2

        shift
        /opt/kera/install/bin/server "$@" 1>$HOME/server-${SLURM_PROCID}.out 2>&1 &
        status=$?
        if [ $status -ne 0 ]; then
            echo "Failed to start plasma: $status"
            exit $status
        fi

        exec tail -f $(ls -Art $HOME/plasma-${SLURM_PROCID}.out | tail -n 1) & tail -f $(ls -Art $HOME/server-${SLURM_PROCID}.out | tail -n 1)

        fi
    ;;
esac

```

### Step 3: Create the singularity container

* Either directly create the kera.sif singularity container, or use a sandbox to eventually modify/add before exporting to sif format

```bash
#directly create a singularity container
sudo singularity build kera.sif docker-daemon://kera:latest
#create the sandbox directory from existing docker kera container, then create the kera.sif
sudo singularity build --sandbox keraUbuntu1804 docker-daemon://kera:latest
sudo singularity build kera.sif keraUbuntu1804/
```

### Step 4: Create a script to install KerA on Iris

* The following script runKerA.sh runs singularity kera.sif container for creating 1 coordinator and 2 brokers, each on a container instance

```bash
#!/bin/bash -l
#SBATCH -J Singularity_KerA_Coord
#SBATCH -N 3 # Nodes
#SBATCH -n 3 # Tasks
#SBATCH --ntasks-per-node=1
#SBATCH --mem=9GB
#SBATCH -c 3 # Cores assigned to each task
#SBATCH --time=0-00:15:00
#SBATCH -p batch
#SBATCH --qos=normal
#SBATCH --mail-user=firstname.lastname@uni.lu
#SBATCH --mail-type=BEGIN,END

module load tools/Singularity

hostName="$(facter hostname)-ib0"
IP=$(facter ipaddress_ib0)
echo "On your laptop: ssh -p 8022 -NL 8889:$hostName:8889 ${USER}@access-iris.uni.lu"

echo "SLURM_JOBID  = ${SLURM_JOBID}"
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST}"
echo "SLURM_NNODES = ${SLURM_NNODES}"
echo "SLURM_NTASK  = ${SLURM_NTASKS}"
echo "Submission directory = ${SLURM_SUBMIT_DIR}"

export KERA_WORKER_CORES=${SLURM_CPUS_PER_TASK:-1}

echo "Cores: $KERA_WORKER_CORES"

export DAEMON_MEM=${SLURM_MEM_PER_CPU:=2048}
export KERA_MEM=$(( ${DAEMON_MEM}*${KERA_WORKER_CORES} ))

export KERA_MASTER_HOST=$(hostname -s)

export NWLOCATOR="infrc:host"

#srun --exclusive -N 1 -n 1 -l -o $HOME/coordout \ 
 singularity run --bind /dev/infiniband,/etc/libibverbs.d  kera.sif \
 coordinator -C $NWLOCATOR=${hostName},port=11100,dev=mlx5_0 --maxCores ${SLURM_CPUS_PER_TASK:-2} -n --reset --clusterName test &

pid=$!
sleep 10s

echo "Starting brokers"

KERA_BROKER_LAUNCHER=${HOME}/kera-start-brokers-${SLURM_JOBID}.sh
echo " - create broker launcher script '${KERA_BROKER_LAUNCHER}'"
cat << 'EOF' > ${KERA_BROKER_LAUNCHER}
#!/bin/bash

echo "I am ${SLURM_PROCID} running on:"
hostname

KERA_WORKER_CORES=${SLURM_CPUS_PER_TASK:-1}

echo "Cores: $KERA_WORKER_CORES"

DAEMON_MEM=${SLURM_MEM_PER_CPU:=2048}
KERA_MEM=$(( ${DAEMON_MEM}*${KERA_WORKER_CORES} ))

echo "memory: $KERA_MEM"

KERA_WORKER_HOST="`hostname`-ib0"
echo ${KERA_WORKER_HOST}


# --bind /opt/mellanox,/sys/class/infiniband \
#  --bind /sys/class/infiniband_cm,/sys/class/infiniband_mad \
#  --bind /sys/class/infiniband_verbs

singularity run --bind /dev/infiniband,/etc/libibverbs.d kera.sif \
 broker -L $NWLOCATOR=${KERA_WORKER_HOST},port=11101,dev=mlx5_0 --totalMasterMemory ${KERA_MEM} \
 -C $NWLOCATOR=$1,port=11100,dev=mlx5_0 --cleanerBalancer fixed:50 -D -d --detectFailures 0 -h 1 \
 -f /tmp/storagemaster1 --maxCores ${SLURM_CPUS_PER_TASK:-2} --clusterName test -r 0 --masterOnly \
 --numberActiveGroupsPerStreamlet 1 --masterActiveGroupsPerStreamlet 1

EOF
chmod +x ${KERA_BROKER_LAUNCHER}

# Start the KerA brokers; pass coordinator hostname as param  service ; for srun: --bind /etc/init.d/
srun --exclusive -N 3 -n 3 --ntasks-per-node=1 -l -o $HOME/broker-$(facter hostname).out \
 ${KERA_BROKER_LAUNCHER} ${hostName} &

sleep 900s
wait $pid

echo $HOME

echo "Ready Stopping instance"

```

* Now you can install KerA on Iris. Singularity run will execute the docker-entrypoint.sh which is configured to run on every slurm task except the one running the coordinator.

```bash
Login to Iris
sbatch runKerA.sh
```

### Step 5: How the output looks

* The Coordinator (coordinator.out)

```bash
1629490163.314847961 CoordinatorMain.cc:110 in main NOTICE[1]: Command line: /opt/kera/install/bin/coordinator -C infrc:host=iris-079-ib0,port=11100,dev=mlx5_0 --maxCores 3 -n --reset --clusterName test
1629490163.314866331 CoordinatorMain.cc:111 in main NOTICE[1]: Coordinator process id: 23390
1629490163.422875345 Infiniband.h:106 in DeviceList WARNING[1]: identified infiniband device: mlx5_0
1629490163.422888057 Infiniband.h:117 in lookup WARNING[1]: looking to open infiniband device: mlx5_0 searching mlx5_0
1629490163.442540847 InfRcTransport.cc:263 in InfRcTransport NOTICE[1]: InfRc listening on UDP: 172.19.6.79:11100
1629490163.444650651 InfRcTransport.cc:272 in InfRcTransport NOTICE[1]: Local Infiniband lid is 108
1629490163.621857932 CoordinatorMain.cc:122 in main NOTICE[1]: coordinator: Listening on infrc:host=iris-079-ib0,port=11100,dev=mlx5_0
1629490163.621874294 CoordinatorMain.cc:125 in main NOTICE[1]: PortTimeOut=-1
1629490163.621876873 PortAlarm.cc:174 in setPortTimeout NOTICE[1]: Set PortTimeout to -1 (ms: -1 to disable.)
1629490163.621883782 CoordinatorMain.cc:146 in main WARNING[1]: Reset requested: deleting external storage for workspace '/ramcloud/test'
1629490163.621891380 CoordinatorMain.cc:151 in main NOTICE[1]: Cluster name is 'test', external storage workspace is '/ramcloud/test/'
1629490163.623065380 CoordinatorClusterClock.cc:170 in recoverClusterTime WARNING[1]: couldn't find "coordinatorClusterClock" object in external storage; starting new clock from zero; benign if starting new cluster from scratch, may cause linearizability failures otherwise
1629490163.623074634 CoordinatorClusterClock.cc:176 in recoverClusterTime NOTICE[1]: initializing CoordinatorClusterClock: startingClusterTime = 0
1629490163.624238702 CoordinatorUpdateManager.cc:82 in init WARNING[7]: couldn't find "coordinatorUpdateManager" object in external storage; starting new cluster from scratch
1629490163.625497673 CoordinatorServerList.cc:412 in recover NOTICE[7]: CoordinatorServerList recovery completed: 0 master(s), 0 backup(s), 0 update(s) to disseminate, server list version is 0
1629490163.625508879 TableManager.cc:808 in recover NOTICE[7]: Table recovery complete: 0 table(s)
1629490163.625516433 CoordinatorService.cc:125 in init NOTICE[7]: Coordinator state has been recovered from external storage; starting service
1629490163.627406243 MemoryMonitor.cc:76 in handleTimerEvent NOTICE[8]: Memory usage now 676 MB (increased 676 MB)
1629490180.817172953 CoordinatorServerList.cc:160 in enlistServer NOTICE[5]: Enlisting server at infrc:host=iris-080-ib0,port=11101,dev=mlx5_0 (server id 1.0) supporting services: MASTER_SERVICE, ADMIN_SERVICE
1629490181.047367269 CoordinatorServerList.cc:160 in enlistServer NOTICE[5]: Enlisting server at infrc:host=iris-082-ib0,port=11101,dev=mlx5_0 (server id 2.0) supporting services: MASTER_SERVICE, ADMIN_SERVICE

```

* The Brokers (server-1.out, server-2.out)

```bash
1629490175.584982556 ServerMain.cc:289 in main NOTICE[1]: Command line: /opt/kera/install/bin/server -L infrc:host=iris-080-ib0,port=11101,dev=mlx5_0 --totalMasterMemory 6144 -C infrc:host=iris-079-ib0,port=11100,dev=mlx5_0 --cleanerBalancer fixed:50 -D -d --detectFailures 0 -h 1 -f /tmp/storagemaster1 --maxCores 3 --clusterName test -r 0 --masterOnly --numberActiveGroupsPerStreamlet 1 --masterActiveGroupsPerStreamlet 1
1629490175.585006261 ServerMain.cc:290 in main NOTICE[1]: Server process id: 22528
1629490175.633273359 Infiniband.h:106 in DeviceList WARNING[1]: identified infiniband device: mlx5_0
1629490175.633290399 Infiniband.h:117 in lookup WARNING[1]: looking to open infiniband device: mlx5_0 searching mlx5_0
1629490175.650759076 InfRcTransport.cc:263 in InfRcTransport NOTICE[1]: InfRc listening on UDP: 172.19.6.80:11101
1629490175.652109813 InfRcTransport.cc:272 in InfRcTransport NOTICE[1]: Local Infiniband lid is 110
1629490175.824960501 ServerMain.cc:319 in main NOTICE[1]: MASTER_SERVICE, ADMIN_SERVICE: Listening on infrc:host=iris-080-ib0,port=11101,dev=mlx5_0
1629490175.825434143 ServerMain.cc:360 in main NOTICE[1]: Using 0 backups
1629490175.825442948 ServerConfig.h:619 in setLogAndHashTableSize NOTICE[1]: Master to allocate 6442450944 bytes total, 1048576 of which are for the hash table
1629490175.825444554 ServerConfig.h:621 in setLogAndHashTableSize NOTICE[1]: Master will have 767 segments and 16384 lines in the hash table
1629490175.825445520 ServerConfig.h:625 in setLogAndHashTableSize NOTICE[1]: Hash table will have one entry for every 49144 bytes in the log
1629490175.825460967 ServerMain.cc:365 in main NOTICE[1]: PortTimeOut=-1
1629490175.825462037 PortAlarm.cc:174 in setPortTimeout NOTICE[1]: Set PortTimeout to -1 (ms: -1 to disable.)
1629490175.825756452 MemoryMonitor.cc:76 in handleTimerEvent NOTICE[4]: Memory usage now 647 MB (increased 647 MB)
1629490175.825781222 Server.cc:101 in run NOTICE[1]: Starting services
1629490175.825795738 Server.cc:165 in createAndRegisterServices NOTICE[1]: Starting master service
1629490175.825796613 Server.cc:166 in createAndRegisterServices NOTICE[1]: Master is using 0 backups
1629490175.825849161 LargeBlockOfMemory.h:255 in mmapGigabyteAligned NOTICE[1]: Populating pages; progress 0 of 6143 MB
1629490176.442748331 LargeBlockOfMemory.h:255 in mmapGigabyteAligned NOTICE[1]: Populating pages; progress 1024 of 6143 MB
1629490177.050605836 LargeBlockOfMemory.h:255 in mmapGigabyteAligned NOTICE[1]: Populating pages; progress 2048 of 6143 MB
1629490177.636708533 LargeBlockOfMemory.h:255 in mmapGigabyteAligned NOTICE[1]: Populating pages; progress 3072 of 6143 MB
1629490178.230278663 LargeBlockOfMemory.h:255 in mmapGigabyteAligned NOTICE[1]: Populating pages; progress 4096 of 6143 MB
1629490178.811673036 LargeBlockOfMemory.h:255 in mmapGigabyteAligned NOTICE[1]: Populating pages; progress 5120 of 6143 MB
1629490179.396368849 SegletAllocator.cc:171 in initializeEmergencyHeadReserve NOTICE[1]: Reserved 2 seglets for emergency head segments (16 MB). 765 seglets (6120 MB) left in default pool.
1629490179.657004084 InfRcTransport.h:118 in registerMemory NOTICE[1]: Registered 6441402368 bytes at 0x40000000
1629490179.657069138 LargeBlockOfMemory.h:255 in mmapGigabyteAligned NOTICE[1]: Populating pages; progress 0 of 1 MB
1629490179.658002890 SegletAllocator.cc:206 in initializeCleanerReserve NOTICE[1]: Reserved 1 seglets for the cleaner (8 MB). 764 seglets (6112 MB) left in default pool.
1629490179.658010446 LogCleaner.cc:898 in FixedBalancer NOTICE[1]: Using fixed balancer with 50% disk cleaning
1629490179.659099100 MultiFileStorage.cc:1063 in MultiFileStorage NOTICE[1]: Backup storage opened with 4294967296 bytes available; allocated 512 frame(s) across 1 file(s) with 8388608 bytes per frame
1629490179.740875053 BackupStorage.cc:82 in benchmark NOTICE[1]: Backup storage speeds (min): 1251 MB/s read
1629490179.740897688 BackupStorage.cc:83 in benchmark NOTICE[1]: Backup storage speeds (avg): 1613 MB/s read,
1629490179.740898718 BackupStorage.cc:89 in benchmark NOTICE[1]: RANDOM_REFINE_AVG BackupStrategy selected
1629490179.740931574 MultiFileStorage.cc:1556 in tryLoadSuperblock NOTICE[1]: Stored superblock had a bad checksum: stored checksum was 0, but stored data had checksum 88a5c087
1629490179.740935386 MultiFileStorage.cc:1556 in tryLoadSuperblock NOTICE[1]: Stored superblock had a bad checksum: stored checksum was 0, but stored data had checksum 88a5c087
1629490179.740936381 MultiFileStorage.cc:1342 in loadSuperblock WARNING[1]: Backup couldn't find existing superblock; starting as fresh backup.
1629490179.740939997 PersistenceManager.cc:88 in PersistenceManager NOTICE[1]: Backup storing replicas with clusterName 'test'. Future backups must be restarted with the same clusterName for replicas stored on this backup to be reused.
1629490179.740941776 PersistenceManager.cc:102 in PersistenceManager NOTICE[1]: Replicas stored on disk have a different clusterName ('__unnamed__'). Scribbling storage to ensure any stale replicas left behind by old backups aren't used by future backups
socket() suceeded for pathname /tmp/plasma
Socket pathname is ok.
socket_fd: 11
1629490179.888345368 Server.cc:168 in createAndRegisterServices NOTICE[1]: Master service started
1629490179.888356298 Server.cc:180 in createAndRegisterServices NOTICE[1]: Starting admin service
1629490179.888359239 Server.cc:184 in createAndRegisterServices NOTICE[1]: Admin service started
1629490179.888360194 Server.cc:103 in run NOTICE[1]: Services started
1629490179.888361061 Server.cc:108 in run NOTICE[1]: Pinning memory
1629490180.794964742 Server.cc:110 in run NOTICE[1]: Memory pinned
1629490180.795133039 MemoryMonitor.cc:76 in handleTimerEvent NOTICE[4]: Memory usage now 7844 MB (increased 7197 MB)
1629490180.795160786 Server.cc:211 in enlist NOTICE[4]: Enlisting with coordinator
1629490180.813952484 CoordinatorSession.cc:119 in getSession NOTICE[4]: Opened session with coordinator at infrc:host=iris-079-ib0,port=11100,dev=mlx5_0
1629490180.814215878 Server.cc:218 in enlist NOTICE[4]: Enlisted; serverId 1.0
socket() suceeded for pathname /tmp/plasma
Socket pathname is ok.
socket_fd: 12
1629490180.814895298 Server.cc:229 in enlist NOTICE[4]: Created objectServerId 1 ? 1
1629490180.814918048 MasterService.cc:935 in initOnceEnlisted NOTICE[4]: My server ID is 1.0
1629490180.814923772 PersistenceManager.cc:150 in initOnceEnlisted NOTICE[4]: pm My server ID is 1.0
1629490180.821292852 ServerList.cc:200 in applyServerList NOTICE[7]: Server 1.0 is up (server list version 1)
1629490180.836741479 PersistenceManager.cc:156 in initOnceEnlisted NOTICE[4]: PersistenceManager 1.0 will store replicas under cluster name 'test'
1629490181.044582504 ServerList.cc:200 in applyServerList NOTICE[7]: Server 2.0 is up (server list version 2)

```

```bash
1629490175.568762266 ServerMain.cc:289 in main NOTICE[1]: Command line: /opt/kera/install/bin/server -L infrc:host=iris-082-ib0,port=11101,dev=mlx5_0 --totalMasterMemory 6144 -C infrc:host=iris-079-ib0,port=11100,dev=mlx5_0 --cleanerBalancer fixed:50 -D -d --detectFailures 0 -h 1 -f /tmp/storagemaster1 --maxCores 3 --clusterName test -r 0 --masterOnly --numberActiveGroupsPerStreamlet 1 --masterActiveGroupsPerStreamlet 1
1629490175.568805199 ServerMain.cc:290 in main NOTICE[1]: Server process id: 4420
1629490175.635965138 Infiniband.h:106 in DeviceList WARNING[1]: identified infiniband device: mlx5_0
1629490175.635992010 Infiniband.h:117 in lookup WARNING[1]: looking to open infiniband device: mlx5_0 searching mlx5_0
1629490175.656650536 InfRcTransport.cc:263 in InfRcTransport NOTICE[1]: InfRc listening on UDP: 172.19.6.82:11101
1629490175.657721727 InfRcTransport.cc:272 in InfRcTransport NOTICE[1]: Local Infiniband lid is 112
1629490175.926597137 ServerMain.cc:319 in main NOTICE[1]: MASTER_SERVICE, ADMIN_SERVICE: Listening on infrc:host=iris-082-ib0,port=11101,dev=mlx5_0
1629490175.927065771 ServerMain.cc:360 in main NOTICE[1]: Using 0 backups
1629490175.927074036 ServerConfig.h:619 in setLogAndHashTableSize NOTICE[1]: Master to allocate 6442450944 bytes total, 1048576 of which are for the hash table
1629490175.927075508 ServerConfig.h:621 in setLogAndHashTableSize NOTICE[1]: Master will have 767 segments and 16384 lines in the hash table
1629490175.927076437 ServerConfig.h:625 in setLogAndHashTableSize NOTICE[1]: Hash table will have one entry for every 49144 bytes in the log
1629490175.927092323 ServerMain.cc:365 in main NOTICE[1]: PortTimeOut=-1
1629490175.927093398 PortAlarm.cc:174 in setPortTimeout NOTICE[1]: Set PortTimeout to -1 (ms: -1 to disable.)
1629490175.930640196 MemoryMonitor.cc:76 in handleTimerEvent NOTICE[5]: Memory usage now 647 MB (increased 647 MB)
1629490175.930684453 Server.cc:101 in run NOTICE[1]: Starting services
1629490175.930690202 Server.cc:165 in createAndRegisterServices NOTICE[1]: Starting master service
1629490175.930691081 Server.cc:166 in createAndRegisterServices NOTICE[1]: Master is using 0 backups
1629490175.930716322 LargeBlockOfMemory.h:255 in mmapGigabyteAligned NOTICE[1]: Populating pages; progress 0 of 6143 MB
1629490176.512783068 LargeBlockOfMemory.h:255 in mmapGigabyteAligned NOTICE[1]: Populating pages; progress 1024 of 6143 MB
1629490177.123410103 LargeBlockOfMemory.h:255 in mmapGigabyteAligned NOTICE[1]: Populating pages; progress 2048 of 6143 MB
1629490177.728711052 LargeBlockOfMemory.h:255 in mmapGigabyteAligned NOTICE[1]: Populating pages; progress 3072 of 6143 MB
1629490178.323984649 LargeBlockOfMemory.h:255 in mmapGigabyteAligned NOTICE[1]: Populating pages; progress 4096 of 6143 MB
1629490178.923948175 LargeBlockOfMemory.h:255 in mmapGigabyteAligned NOTICE[1]: Populating pages; progress 5120 of 6143 MB
1629490179.526665376 SegletAllocator.cc:171 in initializeEmergencyHeadReserve NOTICE[1]: Reserved 2 seglets for emergency head segments (16 MB). 765 seglets (6120 MB) left in default pool.
1629490179.771735859 InfRcTransport.h:118 in registerMemory NOTICE[1]: Registered 6441402368 bytes at 0x40000000
1629490179.771841667 LargeBlockOfMemory.h:255 in mmapGigabyteAligned NOTICE[1]: Populating pages; progress 0 of 1 MB
1629490179.773289133 SegletAllocator.cc:206 in initializeCleanerReserve NOTICE[1]: Reserved 1 seglets for the cleaner (8 MB). 764 seglets (6112 MB) left in default pool.
1629490179.773299815 LogCleaner.cc:898 in FixedBalancer NOTICE[1]: Using fixed balancer with 50% disk cleaning
1629490179.774910848 MultiFileStorage.cc:1063 in MultiFileStorage NOTICE[1]: Backup storage opened with 4294967296 bytes available; allocated 512 frame(s) across 1 file(s) with 8388608 bytes per frame
1629490179.873532112 BackupStorage.cc:82 in benchmark NOTICE[1]: Backup storage speeds (min): 1006 MB/s read
1629490179.873542937 BackupStorage.cc:83 in benchmark NOTICE[1]: Backup storage speeds (avg): 1335 MB/s read,
1629490179.873543851 BackupStorage.cc:89 in benchmark NOTICE[1]: RANDOM_REFINE_AVG BackupStrategy selected
1629490179.873580991 MultiFileStorage.cc:1556 in tryLoadSuperblock NOTICE[1]: Stored superblock had a bad checksum: stored checksum was 0, but stored data had checksum 88a5c087
1629490179.873585051 MultiFileStorage.cc:1556 in tryLoadSuperblock NOTICE[1]: Stored superblock had a bad checksum: stored checksum was 0, but stored data had checksum 88a5c087
1629490179.873586077 MultiFileStorage.cc:1342 in loadSuperblock WARNING[1]: Backup couldn't find existing superblock; starting as fresh backup.
1629490179.873588877 PersistenceManager.cc:88 in PersistenceManager NOTICE[1]: Backup storing replicas with clusterName 'test'. Future backups must be restarted with the same clusterName for replicas stored on this backup to be reused.
1629490179.873590437 PersistenceManager.cc:102 in PersistenceManager NOTICE[1]: Replicas stored on disk have a different clusterName ('__unnamed__'). Scribbling storage to ensure any stale replicas left behind by old backups aren't used by future backups
socket() suceeded for pathname /tmp/plasma
Socket pathname is ok.
socket_fd: 11
1629490180.006412927 Server.cc:168 in createAndRegisterServices NOTICE[1]: Master service started
1629490180.006427969 Server.cc:180 in createAndRegisterServices NOTICE[1]: Starting admin service
1629490180.006431221 Server.cc:184 in createAndRegisterServices NOTICE[1]: Admin service started
1629490180.006432209 Server.cc:103 in run NOTICE[1]: Services started
1629490180.006433238 Server.cc:108 in run NOTICE[1]: Pinning memory
1629490181.028210772 Server.cc:110 in run NOTICE[1]: Memory pinned
1629490181.028383832 MemoryMonitor.cc:76 in handleTimerEvent NOTICE[5]: Memory usage now 7844 MB (increased 7197 MB)
1629490181.028405111 Server.cc:211 in enlist NOTICE[5]: Enlisting with coordinator
1629490181.044703448 CoordinatorSession.cc:119 in getSession NOTICE[5]: Opened session with coordinator at infrc:host=iris-079-ib0,port=11100,dev=mlx5_0
1629490181.044828091 Server.cc:218 in enlist NOTICE[5]: Enlisted; serverId 2.0
socket() suceeded for pathname /tmp/plasma
Socket pathname is ok.
socket_fd: 12
1629490181.045516643 Server.cc:229 in enlist NOTICE[5]: Created objectServerId 2 ? 1
1629490181.045539241 MasterService.cc:935 in initOnceEnlisted NOTICE[5]: My server ID is 2.0
1629490181.045545091 PersistenceManager.cc:150 in initOnceEnlisted NOTICE[5]: pm My server ID is 2.0
1629490181.046144290 PersistenceManager.cc:156 in initOnceEnlisted NOTICE[5]: PersistenceManager 2.0 will store replicas under cluster name 'test'
1629490181.051408793 ServerList.cc:200 in applyServerList NOTICE[6]: Server 1.0 is up (server list version 2)
1629490181.051427966 ServerList.cc:200 in applyServerList NOTICE[6]: Server 2.0 is up (server list version 2)

```
