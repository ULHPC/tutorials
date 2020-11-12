[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/advanced/advanced_scheduling/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/advanced/advanced_scheduling/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# UL HPC Tutorial: HPC Containers with Singularity

     Copyright (c) 2018- UL HPC Team <hpc-sysadmins@uni.lu>


[![](cover_slides.png)](slides.pdf)

<p align="center">
<img src="./images/sing.png" width="300px" >
</p>

## Main Objectives of this Session

* **Discussion on container systems**
    - what they are and where they help
    - common container systems
    - will focus on _Singularity_ container system


* how to use _Singularity_ containers on the UL HPC platform
    -  how to build containers from a definition file
    -  how to import pre-existing containers
    -  how to use applications embedded in containers
* containerized parallel applications execution

## A brief intro. to containers

### Purpose of containers?

* **Application portability**
    - containers bundle together an entire runtime env. (OS to apps.)
    - easy replication of environments
* Services isolation
    - separate microservices in different containers
* Do more with less
    - fast instantiation and tear-down
    - little memory/CPU overhead



### Technology main points

* OS-level virtualization - _light virtualization_
    - don't spin up a full virtual machine
* Close to native _bare metal_ speed

### Common container systems

* [**Docker**](https://www.docker.com)
    - A new (2013-) take on containers (OpenVZ and LXC came before)
    - High uptake in Enterprise (microservices) & science (reproducibility)
    - In use everywhere (esp. DevOps), available on most Cloud infra.

* [**Shifter**](https://github.com/NERSC/shifter)
    - Linux containers for HPC, developed at NERSC
    - Uses Docker functionality but makes it safe in shared HPC systems
    - Image gateway used to convert Docker images before use

* [**Singularity**](https://github.com/sylabs/singularity)
    - Containers for science, initially developed at LBNL
    - Not based on Docker, but can directly import/run Docker images
    - Also HPC oriented, diff. take to running MPI software than Shifter
    - Provides an [Image Registry]{https://github.com/singularityhub/sregistry}


### Singularity in a nutshell


<p align="center">
<img src="./images/singularity_workflow2.png" width="900px" >
</p>

* **build environment**: your workstation (admin. required)
* **production environmemnt**: [UL HPC clusters](https://hpc.uni.lu/systems/clusters.html)

Source: [Kurtzer GM, Sochat V, Bauer MW (2017) Singularity: Scientific containers for mobility of compute. PLoS ONE 12(5): e0177459](https://doi.org/10.1371/journal.pone.0177459)

## Singularity setups

### Build env - Debian/Ubuntu

* Install dependencies

```{.bash}
sudo apt-get update && sudo apt-get install -y build-essential \
     libssl-dev uuid-dev libgpgme11-dev squashfs-tools \
     libseccomp-dev wget pkg-config git cryptsetup
```

* Installing [go](https://sylabs.io/guides/3.6/user-guide/quick_start.html#install)

```bash
export VERSION=1.15 OS=linux ARCH=amd64 && \
  wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz && \
  sudo tar -C /usr/local -xzvf go$VERSION.$OS-$ARCH.tar.gz && \
  rm go$VERSION.$OS-$ARCH.tar.gz

echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc && \
source ~/.bashrc
```

* Download singularity [source](https://sylabs.io/guides/3.6/user-guide/quick_start.html#download)

```bash
export VERSION=3.6.4
export URL="https://github.com/sylabs/singularity/releases/download"
wget "${URL}/v${VERSION}/singularity-${VERSION}.tar.gz"
tar -xzf singularity-${VERSION}.tar.gz
cd singularity
```

* Compiling singularity from source

```bash
./mconfig && \
    make -C builddir && \
    sudo make -C builddir install
```


### Build env - CentOS & RHEL
  - The epel (Extra Packages for Enterprise Linux) repos contain Singularity
  - The singularity package is actually split into two packages called `singularity-runtime`
  - The package `singularity` which also gives you the ability to build Singularity containers

```bash
sudo yum update -y
sudo yum install -y epel-release
sudo yum update -y
sudo yum install -y singularity-runtime singularity
```

See also: https://sylabs.io/guides/3.0/user-guide/installation.html#install-on-linux

### Build env - macOS

* Prerequisites - install Brew, VirtualBox and Vagrant

```bash
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew cask install virtualbox
brew cask install vagrant
brew cask install vagrant-manager
```
* Initialize an Ubuntu VM and install Singularity inside

```bash
mkdir singularity-vm && cd singularity-vm
export VM=sylabs/singularity-3.2-ubuntu-bionic64
vagrant init $VM
vagrant up
vagrant ssh
```

See also: https://sylabs.io/guides/3.0/user-guide/installation.html#install-on-windows-or-mac

### Use on the UL HPC clusters

- Production environment (no admin rights)
- Images can be pulled from official repository (dockerhub,shub,...)
- Images **CANNOT** be built on the ULHPC platform

```bash
module load tools/Singularity
```

### Now that Singularity is there...

```bash
singularity
Usage:
  singularity [global options...] <command>
Available Commands:
  build       Build a Singularity image
  cache       Manage the local cache
  capability  Manage Linux capabilities for users and groups
  config      Manage various singularity configuration (root user only)
  delete      Deletes requested image from the library
  exec        Run a command within a container
  inspect     Show metadata for an image
  instance    Manage containers running as services
  key         Manage OpenPGP keys
  oci         Manage OCI containers
  plugin      Manage Singularity plugins
  pull        Pull an image from a URI
  push        Upload image to the provided URI
  remote      Manage singularity remote endpoints
  run         Run the user-defined default command within a container
  run-help    Show the user-defined help for an image
  search      Search a Container Library for images
  shell       Run a shell within a container
  sif         siftool is a program for Singularity Image Format (SIF) file manipulation
  sign        Attach digital signature(s) to an image
  test        Run the user-defined tests within a container
  verify      Verify cryptographic signatures attached to an image
  version     Show the version for Singularity
```

## Quick start with Singularity


### Pulling from DockerHub

* You can pull images from [DockerHub](https://hub.docker.com/)
  - Example for a specific python version

```bash
singularity pull docker://python:3.8.0b1-alpine3.9
singularity exec python_3.8.0b1-alpine3.9.sif python3
singularity shell python_3.8.0b1-alpine3.9.sif
```
* The ouput is the following:

```bash
./python_3.8.0b1-alpine3.9.sif 
Python 3.8.0b1 (default, Jun  5 2019, 23:34:27) 
[GCC 8.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> print("Running python from within the container.") 
```

* This brought us an immutable image with (tiny) Alpine Linux & Python 3.8.0b1 from the Docker Registry.
<span style="color:red">The image is not writeable, but has access to our home directory by default.</span>

### Building from scratch

* **Sandbox mode**: container development in your build env (laptop)

```bash
sudo singularity build --sandbox \
     python_3.7.3-stretch docker://python:3.7.3-stretch
sudo singularity exec --writable \
     python_3.7.3-stretch/ pip3 install numpy nose test
singularity exec python_3.7.3-stretch \
     python3 -c "import numpy; numpy.test()"
```

This time the Docker Image was downloaded and unpacked to a directory (sandbox mode).
<span style="color:red">Changes within the directory can be made persistent with the **writable** flag.</span>

* Building a **sif image** from the sandbox 
  - done when ready to use envrionment in production
  - just have to send it to your dedicated space

```bash
sudo singularity build python_3.7.3-stretch.sif python_3.7.3-stretch/ 
```

Now image can be transferred, e.g. to the Iris cluster and used normally.

* Containers' access to filesystem(s)
  - Home directories are bind mounted by default
  - Your user(name) and group(s) are dynamically added
  - thus files created maintain normal permissions
  - Other paths need to be explicitly set

```bash
# Build the image on your laptop
sudo singularity build custom.sif python_3.7.3-stretch/
# Test it on you laptop
singularity exec --bind /work/projects/myprj/:/mnt \
            custom.sif python3 /mnt/my_nice_code.py
singularity exec --bind /work/projects/myprj:/work/projects/myprj \
            --bind /scratch/users/$USER:/scratch/users/$USER \
            custom.sif python3 /work/projects/myprj/nice_code.py -o \
	        /scratch/users/$USER/output_dir/
```

With the first command we create a compressed, *SIF - Singularity Image File* from the sandbox folder.
Then, we run the python3 interpreter from this image on code and data existing outside the container.

More details on SIF: https://archive.sylabs.io/2018/03/sif-containing-your-containers/


### Definition headers

* **%setup**: commands in the `%setup` section are first executed on the host system outside of the container after the base OS has been installed
* **%files**: the `%files` section allows you to copy files into the container 
* **%app***: redundant to build different containers for each app with nearly equivalent dependencies
* **%post**: install new software and libraries, write configuration files, create new directories
* **%test**: the `%test` section runs at the very end of the build process to validate the container using a method of your choice
* **%environment**: The `%environment` section allows you to define environment variables that will be set at runtime
* **%startscript**: the contents of the %startscript section are written to a file within the container at build time. This file is executed when the `instance start` command is issued
* **%runscript**: the contents of the `%runscript` section are written to a file within the container that is executed when the container image is run (either via the `singularity run` command or by executing the container directly as a command
* **%labels**: the %labels section is used to add metadata to the file /.singularity.d/labels.json within your container. The general format is a name-value pair
* **%help**: any text in the `%help` section is transcribed into a metadata file in the container during the build. This text can then be displayed using the run-help command

### Applications 

```bash
Bootstrap: docker
From: ubuntu

%environment
    GLOBAL=variables
    AVAILABLE="to all apps"
##############################
# foo
##############################
%apprun foo
    exec echo "RUNNING FOO"
%applabels foo
   BESTAPP FOO
%appinstall foo
   touch foo.exec
%appenv foo
    SOFTWARE=foo
    export SOFTWARE
%apphelp foo
    This is the help for foo.
%appfiles foo
   foo.txt
##############################
# bar
##############################
%apphelp bar
    This is the help for bar.
%applabels bar
   BESTAPP BAR
%appinstall bar
    touch bar.exec
%appenv bar
    SOFTWARE=bar
    export SOFTWARE
```
* Add %app prefix to headers and finish with the application name
* Ex: `singularity run --app foo my_container.sif`


## Advanced Singularity 

* Objectives:
  - Provide a Jupyter notebook container with full features
  - features: IPython Parallel, Virtualenv, CUDA, MPI 

### Step 1: Jupyter

* We will consider the next singularity definition file

```bash
Bootstrap: library
From: ubuntu:18.04
Stage: build
%setup
    touch /file_on_host
    touch ${SINGULARITY_ROOTFS}/file_on_guest
%files
    /file_on_host /opt
%environment
    export PORT=8889
    export LC_ALL=C
%post
    apt-get install -y software-properties-common
    add-apt-repository multiverse
    apt-get update
    apt-get install -y python3 python3-pip python3-venv
    python3 -m pip install jupyter
```

```bash    
%runscript
    echo "Container was created $NOW"
    echo "Arguments received: $*"
    exec python3 "$@"
%startscript
    echo "Started new instance on $(date)"
%test
    grep -q NAME=\"Ubuntu\" /etc/os-release
    if [ $? -eq 0 ]; then
        echo "Container base is Ubuntu as expected."
    else
        echo "Container base is not Ubuntu."
    fi
    python3 -m pip show jupyter
%labels
    Author ekieffer
    Version v0.0.1
%help
    This is a demo container used to illustrate a def
    file that uses all supported sections.
```

```bash 
sudo singularity build jupyter.sif jupyter.def
rsync -avz jupyter.def iris-cluster:jupyter.sif # to the cluster

```

* Next, we need to prepare a launcher ...

```bash
#!/bin/bash -l
#SBATCH -J Singularity_Jupyter
#SBATCH -N 1 # Nodes
#SBATCH -n 1 # Tasks
#SBATCH -c 2 # Cores assigned to each task
#SBATCH --time=0-01:00:00
#SBATCH -p batch
#SBATCH --qos=normal
#SBATCH --mail-user=<firstname>.<lastname>@uni.lu
#SBATCH --mail-type=BEGIN,END

module load tools/Singularity
# Avoid to modify your current jupyter config
export JUPYTER_CONFIG_DIR="$HOME/jupyter_sing/$SLURM_JOBID/"
export JUPYTER_PATH="$HOME/jupyter_sing/$SLURM_JOBID/jupyter_path"
export JUPYTER_DATA_DIR="$HOME/jupyter_sing/$SLURM_JOBID/jupyter_data"
export JUPYTER_RUNTIME_DIR="$HOME/jupyter_sing/$SLURM_JOBID/jupyter_runtime"
export IPYTHONDIR="$HOME/ipython_sing/$SLURM_JOBID"
mkdir -p $IPYTHONDIR

IP=$(facter ipaddress)
echo "On your laptop: ssh -p 8022 -NL 8889:$IP:8889 ${USER}@access-iris.uni.lu " 
singularity instance start jupyter.sif jupyter
singularity exec instance://jupyter jupyter \
    notebook --ip $(facter ipaddress) --no-browser --port 8889 &
pid=$!
sleep 5s
singularity exec instance://jupyter  jupyter notebook list
wait $pid
echo "Stopping instance"
singularity instance stop jupyter
```

To start the container job: `sbatch <launcher_name>.sh`

* Note that the number of cpus reported by python itself will be 28 (regular nodes)
* This is obvisouly wrong. Check  `$SLURM_CPUS_PER_TASK`
* Never use `os.cpu_count()` nor `multiprocessing.cpu_count()`

<p align="center">
<img src="./images/jupyter_cores.png" width="900px" >
</p>

### Step 2:  Jupyter + custom Kernels

* One can wonder **WHY** we want to work with venv in a singularity image:
    - if you want to install **mpi4py**, it is better to do on the cluster to link to the offcial module
    - Since you can't build container on the ULHPC, you can only use a venv


* We are going to use the same jupyter.sif image created before

```bash
#!/bin/bash -l
#SBATCH -J Singularity_Jupyter
#SBATCH -N 1 # Nodes
#SBATCH -n 1 # Tasks
#SBATCH -c 2 # Cores assigned to each tasks
#SBATCH --time=0-01:00:00
#SBATCH -p batch
#SBATCH --qos=normal
#SBATCH --mail-user=<firstname>.<lastname>@uni.lu
#SBATCH --mail-type=BEGIN,END

module load tools/Singularity
export VENV="$HOME/.envs/venv"
export JUPYTER_CONFIG_DIR="$HOME/jupyter_sing/$SLURM_JOBID/"
export JUPYTER_PATH="$VENV/share/jupyter":"$HOME/jupyter_sing/$SLURM_JOBID/jupyter_path"
export JUPYTER_DATA_DIR="$HOME/jupyter_sing/$SLURM_JOBID/jupyter_data"
export JUPYTER_RUNTIME_DIR="$HOME/jupyter_sing/$SLURM_JOBID/jupyter_runtime"
export IPYTHONDIR="$HOME/ipython_sing/$SLURM_JOBID"
mkdir -p $JUPYTER_CONFIG_DIR
mkdir -p $IPYTHONDIR

echo "On your laptop: ssh -p 8022 -NL 8889:$(facter ipaddress):8889 ${USER}@access-iris.uni.lu " 

singularity instance start jupyter.sif jupyter

if [ ! -d "$VENV" ];then
    singularity exec instance://jupyter python3 -m venv $VENV --system-site-packages
    singularity exec instance://jupyter bash -c "source $VENV/bin/activate && python3 -m ipykernel install --sys-prefix --name HPC_SCHOOL_ENV --display-name HPC_SCHOOL_ENV"
fi

singularity exec instance://jupyter bash -c "source $VENV/bin/activate && jupyter \
    notebook --ip $(facter ipaddress) --no-browser --port 8889" &
pid=$!
sleep 5s
singularity exec instance://jupyter bash -c "source $VENV/bin/activate &&  jupyter notebook list"
singularity exec instance://jupyter bash -c "source $VENV/bin/activate &&  jupyter --paths"
singularity exec instance://jupyter bash -c "source $VENV/bin/activate &&  jupyter kernelspec list"

wait $pid
echo "Stopping instance"
singularity instance stop jupyter
```






