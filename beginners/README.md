[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/beginners/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/beginners/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/beginners/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# Getting Started on the UL HPC platform

     Copyright (c) 2013-2018 UL HPC Team <hpc-sysadmins@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/beginners/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/beginners/slides.pdf)


This tutorial will guide you through your first steps on the
[UL HPC platform](http://hpc.uni.lu).

Before proceeding:

* make sure you have an account (if not, follow [this procedure](https://hpc.uni.lu/get_an_account)), and an SSH client.
* take a look at the [quickstart guide](https://hpc.uni.lu/users/quickstart.html)
* ensure you operate from a Linux / Mac environment. Most commands below assumes running in a Terminal in this context. If you're running Windows, you can use MobaXterm, Putty tools etc. as described [on this page](https://hpc.uni.lu/users/docs/access/access_windows.html) yet it's probably better that you familiarize "natively" with Linux-based environment by having a Linux Virtual Machine (consider for that [VirtualBox](https://www.virtualbox.org/)).

From a general perspective, the [Support page](https://hpc.uni.lu/users/docs/report_pbs.html) describes how to get help during your UL HPC usage.

**Convention**

In the below tutorial, you'll proposed terminal commands where the prompt is denoted by `$>`.
M
In general, we will prefix to precise the execution context (_i.e._ your laptop, a cluster frontend or a node). Remember that `#` character is a comment. Example:

		# This is a comment
		$> hostname

		(laptop)$> hostname         # executed from your personal laptop / workstation

		(access-iris)$> hostname    # executed from access server of the Iris cluster


## Platform overview.

You can find a brief overview of the platform with key characterization numbers [on this page](https://hpc.uni.lu/systems/overview.html).

The general organization of each cluster is depicted below:

![UL HPC clusters general organization](https://hpc.uni.lu/images/overview/clusters_general_organization.png)

Details on this organization can be found [here](https://hpc.uni.lu/systems/clusters.html#clusters-organization)

## Hands-On/SSH & UL HPC access


* [Access / SSH Tutorial](https://hpc.uni.lu/users/docs/access.html)

The way SSH handles the keys and the configuration files is illustrated in the following figure:

![SSH key management](https://hpc.uni.lu/images/docssh/schema.png)

In order to be able to login to the clusters, you have sent us through the Account request form the **public key** (i.e. `id_rsa.pub` or the **public key** as saved by MobaXterm/PuttY) you initially generated, enabling us to configure the `~/.ssh/authorized_keys` file of your account.


### Step 1a: Connect to UL HPC (Linux / Mac OS / Unix)

Run the following commands in a terminal (substituting *yourlogin* with the login name you received from us):

        (laptop)$> ssh -p 8022 yourlogin@access-gaia.uni.lu

If you want to connect to the iris cluster,

        (laptop)$> ssh -p 8022 yourlogin@access-iris.uni.lu

Now you probably want to avoid taping this long command to connect to the platform. You can customize SSH aliases for that. Edit the file `~/.ssh/config` (create it if it does not already exist) and adding the following entries:

        Host chaos-cluster
            Hostname access-chaos.uni.lu

        Host gaia-cluster
            Hostname access-gaia.uni.lu

        Host iris-cluster
            Hostname access-iris.uni.lu

        Host *-cluster
            User yourlogin
            Port 8022
            ForwardAgent no

Now you shall be able to issue the following (simpler) command to connect to the cluster and obtain the welcome banner:

		(laptop)$> ssh gaia-cluster

		(laptop)$> ssh iris-cluster

In the sequel, we assume these aliases to be defined.

### Step 1b: Connect to UL HPC (Windows)

* Download [MobaXterm Installer edition](http://mobaxterm.mobatek.net/)
* Install MobaXterm
* Open the application **Start** > **Program Files** > **MobaXterm**
* Change the default home directory for a persistent home directory instead of the default Temp directory. Go onto **Settings** > **Configuration** > **General** > **Persistent home directory**. Choose a location for your home directory.
* load your private SSH key. **Tools** > **Network** > **MobaKeyGen (SSH key generator)** and choose Load (or create a new RSA key).
* click on **Session**
  * In **SSH Session**:
    * Remote host: `access-iris.uni.lu`
		* Check the **Specify username** box
		* Username: `yourlogin`
    * Port: 8022
  * In **Advanced SSH Settings**
	  * Check `Use private key` box
		* Select your previously generated `id_rsa.ppk`
  * Click on **Save**
	* Do the same thing for the other clusters (chaos, gaia) by changing the **Remote host** field.


### Step 2: Connect from one cluster to the other

The SSH key you provided us secure your connection __from__ your laptop (or personal workstation) __to__ the cluster frontends. It is thus important to protect them by a passphrase.

You shall have also a new key pair configured in your account to permit a bi-directional transparent connection from one cluster to the other (you can check that in your `~/.ssh/authorized_keys` and by successfully running:

		(access-gaia)$> ssh chaos-cluster

or

		(access-chaos)$> ssh gaia-cluster

If that's the case, you can ignore the rest of this section.
**Otherwise**, you will now have to configure a passphrase-free SSH key pair to permit a transparent connection from one cluster to another. Have a look at this [FAQ](https://hpc.uni.lu/blog/2017/faq-how-to-permit-bi-directional-connection/)

> If you have some issue to connect to the clusters (for example `Connection closed by remote host` error message), you should check the section on how to [use SSH proxycommand setup to access the clusters despite port filtering](#using-ssh-proxycommand-setup-to-access-the-clusters-despite-port-filtering)

### Hands-on/ Transferring files

Directories such as `$HOME`, `$WORK` or `$SCRATCH` are shared among the nodes of the cluster that you are using (including the front-end) via shared filesystems (NFS, Lustre) meaning that:

* every file/directory pushed or created on the front-end is available on the computing nodes
* every file/directory pushed or created on the computing nodes is available on the front-end


### Step 3a: Linux / OS X / Unix command line tools

The two most common tools you can use for data transfers over SSH:

* `scp`: for the full transfer of files and directories (only works fine for single files or directories of small/trivial size)
* `rsync`: a software application which synchronizes files and directories from one location to another while minimizing data transfer as only the outdated or inexistent elements are transferred (practically required for lengthy complex transfers, which are more likely to be interrupted in the middle).

Of both, normally the second approach should be preferred, as more generic; note that, both ensure a secure transfer of the data, within an encrypted tunnel.

* Create a new directory on your local machine and download a file to transfer (next-gen sequencing data from the NIH Roadmap Epigenomics Project):

		(laptop)$> mkdir file_transfer
		(laptop)$> cd file_transfer
		(laptop)$> wget "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM409nnn/GSM409307/suppl/GSM409307_UCSD.H1.H3K4me1.LL228.bed.gz"

* Transfer the file with scp:

		(laptop)$> scp GSM409307_UCSD.H1.H3K4me1.LL228.bed.gz gaia-cluster:

* Connect to the cluster, check if the file is there and delete it.

		(laptop)$> ssh gaia-cluster
		(access-gaia)$> ls
		(access-gaia)$> rm GSM409307_UCSD.H1.H3K4me1.LL228.bed.gz
		rm: remove regular file `GSM409307_UCSD.H1.H3K4me1.LL228.bed.gz'? y
		(access-gaia)$> exit

* Transfer the directory with rsync:

		(laptop)$> cd ..
		(laptop)$> rsync -avzu file_transfer gaia-cluster:

* Delete the file and retrieve it from the cluster:

		(laptop)$> rm file_transfer/GSM409307_UCSD.H1.H3K4me1.LL228.bed.gz
		(laptop)$> rsync -avzu gaia-cluster:file_transfer .

* **Bonus**: Check where the file is located on the cluster after the rsync.

You can get more information about these transfer methods in the [file transfer documentation](https://hpc.uni.lu/users/docs/filetransfer.html).



### Step 3b: Windows / Linux / OS X / Unix GUI tools

* Download the FileZilla client application from [filezilla-project.org](https://filezilla-project.org/download.php?type=client) and install it.
* First we need to tell FileZilla about our ssh key:
	* Start the application.
	* Go to the `Settings` (either under `Edit` or `FileZilla` depending on the OS).
	* In the category `Connection` select `SFTP`.
	* Click on the button `Add keyfile...` and select your private keyfile (you may need to convert it).
	* Finally click `OK` to save and close the settings.

![Add ssh key](https://github.com/ULHPC/tutorials/raw/devel/basic/getting_started/images/filezilla_key.jpg)

* Back in the main window click on the `Site Manager` button on the top left or select `Site Manager` from the `File` menu.
* Click on the `New Site` button and enter/select the following:
  * Host: `access-gaia.uni.lu`
	* Port: 8022
  * Protocol: `SFTP - SSH File Transfer Protocol`
  * Logon Type: `Interactive`
  * User: your login

![Connection settings](https://github.com/ULHPC/tutorials/raw/devel/basic/getting_started/images/site_manager.jpg)

* Click on the `Connect` button.
* Accept the certificate.

You should now see something similar to the following window:

![Connection settings](https://github.com/ULHPC/tutorials/raw/devel/basic/getting_started/images/filezilla.jpg)

On the very top, beneath the quick connect, you see the message log. Below you have the directory tree and the contents of the current directory for you local computer on the left and the remote location on the right.

To transfer a file, simply drag and drop it from the directory listing on the left side to destination directory on the right (to transfer from local to remote) or vice versa (to transfer from remote to local). You can also select a file by left clicking on it once and then right click on it to get the context menu and select "Upload" or "Download" to transfer it.

If you skipped step 3a, you may download the following file (50 MB) for testing: <br />
[ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM409nnn/GSM409307/suppl/GSM409307_UCSD.H1.H3K4me1.LL228.bed.gz](ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM409nnn/GSM409307/suppl/GSM409307_UCSD.H1.H3K4me1.LL228.bed.gz) (next-gen sequencing data from the NIH Roadmap Epigenomics Project)

When you click the fifth icon on the top with the two green arrows to toggle the transfer queue, you can see the status of ongoing transfers on the very bottom of the window.

![Connection settings](https://github.com/ULHPC/tutorials/raw/devel/basic/getting_started/images/transfer.jpg)

### Step 3c: Windows MobaXterm file transfer

If you are on Windows, you can directly use MobaXterm to transfer files. Connect to your session (see below on how to configure it). On the right panel you should see an **SFTP** panel opened.

![SFTP on MobaXterm](https://github.com/ULHPC/tutorials/raw/devel/basic/getting_started/images/moba_sftp.jpg)

You have just to drag and drop your files to this panel to transfer files to the cluster. You can try to upload this file [ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM409nnn/GSM409307/suppl/GSM409307_UCSD.H1.H3K4me1.LL228.bed.gz](ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM409nnn/GSM409307/suppl/GSM409307_UCSD.H1.H3K4me1.LL228.bed.gz) (next-gen sequencing data from the NIH Roadmap Epigenomics Project)

To retrieve a file from the cluster, you can right click on it and choose the **Download** option. Please refers to MobaXterm documentation for more informations on the available features.

## Discovering, visualizing and reserving UL HPC resources

In the sequel, replace `<login>` in the proposed commands with you login on the platform (ex: `svarrette`).

### Step 1: the working environment

* [reference documentation](http://hpc.uni.lu/users/docs/env.html)

After a successful login onto one of the access node (see [Cluster Access](https://hpc.uni.lu/users/docs/access.html)), you end into your personal homedir `$HOME` which is shared over NFS between the access node and the computing nodes.

Again, remember that your homedir is placed on __separate__ NFS servers on each site, which __ARE NOT SYNCHRONIZED__: data synchronization between each of them remain at your own responsibility. We will see below that the UL HPC team prepared for you a script to facilitate the transfer of data between each site.

Otherwise, you have to be aware of at least two directories:

* `$HOME`: your home directory under NFS.
* `$SCRATCH`: a non-backed up area put if possible under Lustre for fast I/O operations

Your homedir is under a regular backup policy. Therefore you are asked to pay attention to your disk usage __and__ the number of files you store there.

* Estimate file space usage and summarize disk usage of each FILE, recursively for directories using the `ncdu` command:

		(access)$> ncdu

* You shall also pay attention to the number of files in your home directory. You can count them as follows:

		(access)$> find . -type f | wc -l

* You can get an overview of the quotas and your current disk usage with the following command:

		(access)$> df-ulhpc


### Step 2: web monitoring interfaces

Each cluster offers a set of web services to monitor the platform usage:

* A [pie-chart overview of the platform usage](https://hpc.uni.lu/status/overview.html)
* [Monika](https://hpc.uni.lu/status/monika.html), the visualization interface of the OAR scheduler, which  display the status of the clusters as regards the jobs running on the platform.
* [DrawGantt](https://hpc.uni.lu/status/drawgantt.html), the Gantt visualization of jobs scheduled on OAR
* [Ganglia](https://hpc.uni.lu/status/ganglia.html), a scalable distributed monitoring system for high-performance computing systems such as clusters and Grids.

### Step 3a: Reserving resources with Slurm

#### The basics

* [reference documentation](https://hpc.uni.lu/users/docs/slurm.html)

[Slurm](https://slurm.schedmd.com/) Slurm is an open source, fault-tolerant, and highly scalable cluster management and job scheduling system for large and small Linux clusters. It is used on Iris UL HPC cluster.

* It allocates exclusive or non-exclusive access to the resources (compute nodes) to users during a limited amount of time so that they can perform they work
* It provides a framework for starting, executing and monitoring work
* It arbitrates contention for resources by managing a queue of pending work.
* it permits to schedule jobs for users on the cluster resource

There are two types of jobs:

  * _interactive_: you get a shell on the first reserve node
  * _passive_: classical batch job where the script passed as argument to `sbatch` is executed

We will now see the basic commands of Slurm.

* Connect to **iris-cluster**. You can request resources in interactive mode:

		(access)$> srun -p interactive --qos qos-interactive --pty bash

  Notice that with no other parameters, srun gave you one resource for 1 hour. You were also directly connected to the node you reserved with an interactive shell.
  Now exit the reservation:

        (node)$> exit      # or CTRL-D

  When you run exit, you are disconnected and your reservation is terminated.

To avoid anticipated termination of your jobs in case of errors (terminal closed by mistake),
you can reserve and connect in two steps using the job id associated to your reservation.

* First run a passive job _i.e._ run a predefined command -- here `sleep 10d` to delay the execution for 10 days -- on the first reserved node:

		(access)$> sbatch --qos qos-batch --wrap "sleep 10d"
		Submitted batch job 390

  You noticed that you received a job ID (in the above example: `390`), which you can later use to connect to the reserved resource(s):

        (access)$> srun -p interactive --qos qos-interactive --jobid 390 --pty bash # adapt the job ID accordingly ;)
		(node)$> ps aux | grep sleep
		cparisot 186342  0.0  0.0 107896   604 ?        S    17:58   0:00 sleep 1h
		cparisot 187197  0.0  0.0 112656   968 pts/0    S+   18:04   0:00 grep --color=auto sleep
		(node)$> exit             # or CTRL-D

**Question: At which moment the job `390` will end?**

a. after 10 days

b. after 1 hour

c. never, only when I'll delete the job

**Question: manipulate the `$SLURM_*` variables over the command-line to extract the following information, once connected to your job**

a. the list of hostnames where a core is reserved (one per line)
   * _hint_: `man echo`

b. number of reserved cores
   * _hint_: `search for the NPROCS variable`

c. number of reserved nodes
   * _hint_: `search for the NNODES variable`

d. number of cores reserved per node together with the node name (one per line)
   * Example of output:

            12 iris-11
            12 iris-15

   * _hint_: `NPROCS variable or NODELIST`


#### Job management

Normally, the previously run job is still running.

* You can check the status of your running jobs using `squeue` command:

		(access)$> squeue      # access all jobs
		(access)$> squeue -u cparisot  # access all your jobs

  Then you can delete your job by running `scancel` command:

		(access)$> scancel 390


* you can see your system-level utilization (memory, I/O, energy) of a running job using `sstat $jobid`:

		(access)$> sstat 390

In all remaining examples of reservation in this section, remember to delete the reserved jobs afterwards (using `scancel` or `CTRL-D`)

You probably want to use more than one core, and you might want them for a different duration than one hour.

* Reserve interactively 4 tasks with 2 nodes for 30 minutes (delete the job afterwards)

		(access)$> srun -p interactive --qos qos-interactive --time=0:30:0 -N 2 --ntasks-per-node=4 --pty bash


##### Pausing, resuming jobs


To stop a waiting job from being scheduled and later to allow it to be scheduled:

		(access)$> scontrol hold $SLURM_JOB_ID
		(access)$> scontrol release $SLURM_JOB_ID

To pause a running job and then resume it:

		(access)$> scontrol suspend $SLURM_JOB_ID
		(access)$> scontrol resume $SLURM_JOB_ID


### Step 3b: Reserving resources with OAR

#### The basics

* [reference documentation](https://hpc.uni.lu/users/docs/oar.html)

[OAR](http://oar.imag.fr/) is an open-source batch scheduler which provides simple yet flexible facilities for the exploitation of the UL HPC clusters.

* it permits to schedule jobs for users on the cluster resource
* a _OAR resource_ corresponds to a node or part of it (CPU/core)
* a _OAR job_ is characterized by an execution time (walltime) on a set of resources.
  There exists two types of jobs:
  * _interactive_: you get a shell on the first reserve node
  * _passive_: classical batch job where the script passed as argument to `oarsub` is executed **on the first reserved node**

We will now see the basic commands of OAR.

* Connect to one of the UL HPC  frontend. You can request resources in interactive mode:

		(access)$> oarsub -I

  Notice that with no parameters, oarsub gave you one resource (one core) for two hours. You were also directly connected to the node you reserved with an interactive shell.
  Now exit the reservation:

        (node)$> exit      # or CTRL-D

  When you run exit, you are disconnected and your reservation is terminated.

To avoid anticipated termination of your jobs in case of errors (terminal closed by mistake),
you can reserve and connect in two steps using the job id associated to your reservation.

* First run a passive job _i.e._ run a predefined command -- here `sleep 10d` to delay the execution for 10 days -- on the first reserved node:

		(access)$> oarsub "sleep 10d"
		[ADMISSION RULE] Set default walltime to 7200.
		[ADMISSION RULE] Modify resource description with type constraints
		OAR_JOB_ID=919309

  You noticed that you received a job ID (in the above example: `919309`), which you can later use to connect to the reserved resource(s):

        (access)$> oarsub -C 919309        # adapt the job ID accordingly ;)
        Connect to OAR job 919309 via the node e-cluster1-13
		[OAR] OAR_JOB_ID=919309
		[OAR] Your nodes are:
      		e-cluster1-13*1

		(e-cluster1-13)$> java -version
		(e-cluster1-13)$> hostname -f
		(e-cluster1-13)$> whoami
		(e-cluster1-13)$> env | grep OAR   # discover environment variables set by OAR
		(e-cluster1-13)$> exit             # or CTRL-D

**Question: At which moment the job `919309` will end?**

a. after 10 days

b. after 2 hours

c. never, only when I'll delete the job

**Question: manipulate the `$OAR_NODEFILE` variable over the command-line to extract the following information, once connected to your job**

a. the list of hostnames where a core is reserved (one per line)
   * _hint_: `man cat`

b. number of reserved cores (one per line)
   * _hint_: `man wc` --  use `wc -l` over the pipe `|` command

c. number of reserved nodes (one per line)
   * _hint_: `man uniq` -- use `uniq` over the pipe `|` command

d. number of cores reserved per node together with the node name (one per line)
   * Example of output:

    	    12 gaia-11
    	    12 gaia-15

   * _hint_: `man uniq` -- use `uniq -c` over the pipe `|` command

e. **(for geeks)** output the number of reserved nodes times number of cores per node
   * Example of output:

	        gaia-11*12
	        gaia-15*12

   * _hint_: `man awk` -- use `printf` command of `awk` over the pipe command, for instance `awk '{ printf "%s*%d\n",$2,$1 }'`. You might prefer `sed` or any other advanced geek command.

#### Job management

Normally, the previously run job is still running.

* You can check the status of your running jobs using `oarstat` command:

		(access)$> oarstat      # access all jobs
		(access)$> oarstat -u   # access all your jobs

  Then you can delete your job by running `oardel` command:

		(access)$> oardel 919309


* you can see your consumption (in an historical computational measure named _CPU hour_ i.e. the work done by a CPU in one hour of wall clock time) over a given time period using `oarstat --accounting "YYYY-MM-DD, YYYY-MM-DD" -u <youlogin>`:

		(access)$> oarstat --accounting "2016-01-01, 2016-12-31" -u <login>

  In particular, take a look at the difference between the **asked** resources and the **used** ones

In all remaining examples of reservation in this section, remember to delete the reserved jobs afterwards (using `oardel` or `CTRL-D`)

You probably want to use more than one core, and you might want them for a different duration than two hours.
The `-l` switch allows you to pass a comma-separated list of parameters specifying the needed resources for the job.

* Reserve interactively 4 cores for 6 hours (delete the job afterwards)

		(access)$> oarsub -I -l core=6,walltime=6


* Reserve interactively 2 nodes for 3h15 (delete the job afterwards):

		(access)$> oarsub -I -l nodes=3,walltime=3:15

#### Hierarchical filtering of resources

OAR features a very powerful resource filtering/matching engine able to specify resources in a **hierarchical**  way using the `/` delimiter. The resource property hierarchy is as follows:

		enclosure -> nodes -> cpu -> core


*  Reserve interactively 2 cores on 3 different nodes belonging to the same enclosure (**total: 6 cores**) for 3h15:

		(access)$> oarsub -I -l /enclosure=1/nodes=3/core=2,walltime=3:15


* Reserve interactively two full nodes belonging to the different enclosure for 6 hours:

		(access)$> oarsub -I -l /enclosure=2/nodes=1,walltime=6

**Question: reserve interactively 2 cpus on 2 nodes belonging to the same enclosure for 4 hours**

**Question: in the following statements, explain the advantage and drawback (in terms of latency/bandwidth etc.) of each of the proposed approaches**

a. `oarsub -I -l /nodes=2/cpu=1` vs `oarsub -I -l cpu=2` vs `oarsub -I -l /nodes=1/cpu=2`

b. `oarsub -I -l /enclosure=1/nodes=2` vs `oarsub -I -l nodes=2` vs `oarsub -I -l /enclosure=2/nodes=1`

#### Using OAR properties

You might have notice on [Monika](https://hpc.uni.lu/status/monika.html) for each site a list of properties assigned to each resource.

The `-p` switch allows you to specialize (as an SQL syntax) the property you wish to use when selecting the resources. The syntax is as follows: `oarsub -p "< property >='< value >'"`

You can find the available OAR properties on the [UL HPC documentation](https://hpc.uni.lu/users/docs/oar.html#select-nodes-precisely-with-properties). The main ones are described below

|Property        | Description                            | Example                                         |
|----------------|----------------------------------------|-------------------------------------------------|
|host            | Full hostname of the resource          | -p "host='h-cluster1-14.chaos-cluster.uni.lux'" |
|network_address | Short hostname of the resource         | -p "network_address='h-cluster1-14'"            |
|gpu             | GPU availability (gaia only)           | -p "gpu='YES'"                                  |

* reserve interactively 4 cores on a GPU node for 8 hours (_this holds only on the `gaia` cluster_) (**total: 4 cores**)

		(access-gaia)$> oarsub -I -l nodes=1/core=4,walltime=8 -p "gpu='YES'"

* reserve interactively 4 cores on the GPU node `gaia-65` for 8 hours (_this holds only on the `gaia` cluster_) (**total: 4 cores**)

		(access-gaia)$> oarsub -I -l nodes=1/core=4,walltime=8 -p "gpu='yes'" -p "network_address='gaia-65'"


#### Reserving specific resources `bigsmp`and `bigmem`

Some nodes are very specific (for instance the nodes with 1TB of memory or the BCS subsystem of Gaia composed of 4 motherboards of 4 processors with a total of 160 cores aggregated in a ccNUMA architecture).
**Due to this specificity, they are NOT scheduled by default**  and can only be reserved with an explicit oarsub parameter: `-t bigmem` or `-t bigsmp`

* reserve interactively 2 cpu on the bigsmp node belonging to the same board for 3 hours: (**total: 32 cores**)

		(access-gaia)$> oarsub -t bigsmp -I -l /board=1/cpu=2,walltime=3


**Question: why are these resources not scheduled by default?**


#### Reservation at a given period of time

You can use the `-r "YYYY-MM-DD HH:MM:SS"` option of `oarsub` to specify the date you wish the reservation to be issued. This is of particular interest for you to book in advance resources out of the working hours (at night and/or over week ends)


## Hands-on/Using modules

[Environment Modules](http://modules.sourceforge.net/) is a software package that allows us to provide a [multitude of applications and libraries in multiple versions](http://hpc.uni.lu/users/software/) on the UL HPC platform. The tool itself is used to manage environment variables such as `PATH`, `LD_LIBRARY_PATH` and `MANPATH`, enabling the easy loading and unloading of application/library profiles and their dependencies.

We will have multiple occasion to use modules in the other tutorials so there is nothing special we foresee here. You are just encouraged to read the following resources:

* [Introduction to Environment Modules by Wolfgang Baumann](https://www.hlrn.de/home/view/System3/ModulesUsage)
* [Modules tutorial @ NERSC](https://www.nersc.gov/users/software/nersc-user-environment/modules/)
* [UL HPC documentation on modules](https://hpc.uni.lu/users/docs/modules.html)


## Hands-on/Persistent Terminal Sessions using GNU Screen

[GNU Screen](http://www.gnu.org/software/screen/) is a tool to manage persistent terminal sessions.
It becomes interesting since you will probably end at some moment with the following  scenario:

> you frequently program and run computations on the UL HPC platform _i.e_ on a remote Linux/Unix computer, typically working in six different terminal logins to the access server from your office workstation, cranking up long-running computations that are still not finished and are outputting important information (calculation status or results), when you have not 2 interactive jobs running... But it's time to catch the bus and/or the train to go back home.

Probably what you do in the above scenario is to

a. clear and shutdown all running terminal sessions

b. once at home when the kids are in bed, you're logging in again... And have to set up the whole environment again (six logins, 2 interactive jobs etc. )

c. repeat the following morning when you come back to the office.

Enter the long-existing and very simple, but totally indispensable [GNU screen](http://www.gnu.org/software/screen/) command. It has the ability to completely detach running processes from one terminal and reattach it intact (later) from a different terminal login.

### Pre-requisite: screen configuration file `~/.screenrc`

While not mandatory, we advise you to rely on our customized configuration file for screen [`.screenrc`](https://github.com/ULHPC/dotfiles/blob/master/screen/.screenrc) available on [Github](https://github.com/ULHPC/dotfiles/blob/master/screen/.screenrc).
Normally, you have nothing to do since we already setup this file for you in your homedir.
Otherwise, simply clone the [ULHPC dotfile repository](https://github.com/ULHPC/dotfiles/) and make a symbolic link `~/.screenrc` targeting the file `screen/screenrc` of the repository.

### Basic commands

You can start a screen session (_i.e._ creates a single window with a shell in it) with the `screen` command.
Its main command-lines options are listed below:

* `screen`: start a new screen
* `screen -ls`: does not start screen, but prints a list of `pid.tty.host` strings identifying your current screen sessions.
* `screen -r`: resumes a detached screen session
* `screen -x`: attach to a not detached screen session. (Multi display mode _i.e._ when you and another user are trying to access the same session at the same time)


Once within a screen, you can invoke a screen command which consist of a "`CTRL + a`" sequence followed by one other character. The main commands are:

* `CTRL + a c`: (create) creates a new Screen window. The default Screen number is zero.
* `CTRL + a n`: (next) switches to the next window.
* `CTRL + a p`: (prev) switches to the previous window.
* `CTRL + a d`: (detach) detaches from a Screen
* `CTRL + a A`: (title) rename the current window
* `CTRL + a 0-9`: switches between windows 0 through 9.
* `CTRL + a k` or `CTRL + d`: (kill) destroy the current window
* `CTRL + a ?`: (help) display a list of all the command options available for Screen.

### Sample Usage on the UL HPC platform: Kernel compilation

We will illustrate the usage of GNU screen by performing a compilation of a recent linux kernel.

* start a new screen session

        (access)$> screen

* rename the screen window "Frontend" (using `CTRL+a A`)
* create the directory to host the files

		(access)$> mkdir -p PS1/src
		(access)$> cd PS1/src

* create a new window and rename it "Compile"
* within this new window, start a new interactive job over 1 nodes for 4 hours

		(access)$> srun -p interactive --qos qos-interactive --time 4:00:0 -N 1 --pty bash

* detach from this screen (using `CTRL+a d`)
* kill your current SSH connection and your terminal
* re-open your terminal and connect back to the cluster frontend
* list your running screens:

		(access)$> screen -ls
		There is a screen on:
			9143.pts-0.access	(05/04/2014 11:29:43 PM) (Detached)
		1 Socket in /var/run/screen/S-svarrette.

* re-attach your previous screen session

		(access)$> screen -r      # OR screen -r 9143.pts-0.access (see above socket name)

* in the "Compile" windows, go to the working directory and download the Linux kernel sources

		(node)$> cd PS1/src
		(node)$> curl -O https://www.kernel.org/pub/linux/kernel/v3.x/linux-3.13.6.tar.gz

   **IMPORTANT** to ovoid overloading the **shared** file system with the many small files involves in the kernel compilation (_i.e._ NFS and/or Lustre), we will perform the compilation in the **local** file system, _i.e._ either in `/tmp` or (probably more efficient) in `/dev/shm` (_i.e_ in the RAM):

		(node)$> mkdir /dev/shm/PS1
		(node)$> cd /dev/shm/PS1
		(node)$> tar xzf PS1/src/linux-3.13.6.tar.gz
		(node)$> cd linux-3.13.6
		(node)$> make mrproper
		(node)$> make alldefconfig
		(node)$> make 2>&1 | tee /dev/shm/PS1/kernel_compile.log

* You can now detach from the screen and take a coffee

The last compilation command make use of `tee`, a nice tool which read from standard input and write to standard output _and_ files. This permits to save in a log file the message written in the standard output.

**Question: why using the `make 2>&1` sequence in the last command?**

**Question: why working in `/dev/shm` is more efficient?**


* Reattach from time to time to your screen to see the status of the compilation
* Your compilation is successful if it ends with the sequence:

		[...]
		Kernel: arch/x86/boot/bzImage is ready  (#2)

* Restart the compilation, this time using parallel jobs within the Makefile invocation (`-j` option of make)

		(node)$> make clean
		(node)$> time make -j `echo $SLURM_NPROCS` 2>&1 | tee /dev/shm/PS1/kernel_compile.2.log

The table below should convince you to always run `make` with the `-j` option whenever you can...

|   Context                          | time (`make`) | time (`make -j 16`) |
|------------------------------------|---------------|---------------------|
| Compilation in `/tmp`(HDD / chaos) | 4m6.656s      | 0m22.981s           |
| Compilation in `/tmp`(SSD / gaia)  | 3m52.895s     | 0m17.508s           |
| Compilation in `/dev/shm` (RAM)    | 3m11.649s     | 0m17.990s           |


* Use the [Ganglia](https://hpc.uni.lu/status/ganglia.html) interface to monitor the impact of the compilation process on the node your job is running on.
* Use the following system commands on the node during the compilation:

  * `htop`
  * `top`
  * `free -m`
  * `uptime`
  * `ps aux`

## Using a command line text editor

Before the next section, you must learn to use a text editor in command line.
We can recommend `nano` or `vim`: `nano` is very simple, `vim` is complex but very powerful.


### Nano

`$ nano <path/filename>`

* quit and save: `CTRL+x`
* save: `CTRL+o`
* highlight text: `Alt-a`
* Cut the highlighted text: `CTRL+k`
* Paste: `CTRL+u`


### Vim

[`vim <path/filename>`](https://vim.rtorr.com/)

There are 2 main modes:

* Edition mode: press `i` or `insert` once
* Command mode: press `ESC` once

Here is a short list of useful commands:

* save: `:w`
* save and quit: `:wq`
* quit and discard changes: `:q!`
* search: `/<pattern>`
* search & replace: `:%s/<pattern>/<replacement>/g`
* jump to line 100: `:100`
* highlight text: `CTRL+V`
* cut the highlighted text: `d`
* cut one line: `dd`
* paste: `p`
* undo: `u`

## Advanced section
### Using software modules

The UL HPC provides [environment modules](https://hpc.uni.lu/users/docs/modules.html) with the module command
to manage the user environment, e.g. changing the environment variables.

By loading appropriate environment modules, the user can select:

* compilers,
* libraries, e.g. the MPI library, or
* other third party software packages.

An exhaustive list of the available software is proposed [in this page](https://hpc.uni.lu/users/software/).

On a node, using an interactive jobs, you can:

* list all available softwares: `module avail`
* search for one software: `module spider <search terms>`
* "load" a software in your environment: `module load <module name>`
* list the currently loaded modules: `module list`
* clean your environment, unload everything: `module purge`


#### Matlab

1. Create a file named `fibonacci.m` in your home directory, copy-paste the following code in this file.
   This code will calculate the first N numbers of the Fibonacci sequence


        N=1000;
        fib=zeros(1,N);
        fib(1)=1;
        fib(2)=1;
        k=3;
        while k <= N
          fib(k)=fib(k-2)+fib(k-1);
          fprintf('%d\n',fib(k));
          pause(1);
          k=k+1;
        end


2. Create a new interactive job

3. Look for the `matlab` module using the command `module spider`

4. Load the module `base/MATLAB` using the command `module load`

5. Execute the code using matlab

        (node)$> matlab -nojvm -nodisplay -nosplash < path/to/fibonacci.m


#### R

1. Create a file named `fibonacci.R` in your home directory, copy-paste the following code in this file.
   This code will calculate the first N numbers of the Fibonacci sequence


        N <- 130
        fibvals <- numeric(N)
        fibvals[1] <- 1
        fibvals[2] <- 1
        for (i in 3:N) {
             fibvals[i] <- fibvals[i-1]+fibvals[i-2]
             print( fibvals[i], digits=22)
             Sys.sleep(1)
        }

2. Create a new interactive job

3. Look for the `R` module using the command `module spider`

3. Load the module `lang/R` using the command `module load`

4. Execute the code using R

        (node)$> Rscript path/to/fibonacci.R



### Compiling your code

In this section, we will learn to compile small "hello world" programs in different languages, using different compilers and toolchains.

#### C

Create a new file called `helloworld.c`, containing the source code of a simple "Hello World" program written in C.


        #include<stdio.h>

        int main()
        {
            printf("Hello, world!");
            return 0;
        }


First, compile the program using the "FOSS" toochain, containing the GNU C compiler

        (node)$> module load toolchain/foss
        (node)$> gcc helloworld.c -o helloworld

Then, compile the program using the Intel toolchain, containing the ICC compiler

        (node)$> module purge
        (node)$> module load toolchain/intel
        (node)$> icc helloworld.c -o helloworld

If you use Intel CPUs and ICC is available on the platform, it is advised to use ICC in order to produce optimized binaries and achieve better performance.


#### C++

**Question:** create a new file `helloworld.cpp` containing the following C++ source code,
compile the following program, using GNU C++ compiler (`g++` command), and the Intel compiler (`icpc` command).


        #include <iostream>

        int main() {
            std::cout << "Hello, world!" << std::endl;
        }



#### Fortran

**Question:** create a new file `helloworld.f` containing the following source code,
compile the following program, using the GNU Fortran compiler (`gfortran` command), and ICC (`ifortran` command).


        program hello
           print *, "Hello, World!"
        end program hello


Be careful, the 6 spaces at the beginning of each line are required



#### MPI

MPI is a programming interface that enables the communication between processes of a distributed memory system.

We will create a simple MPI program where the MPI process of rank 0 broadcasts an integer (42) to all the other processes.
Then, each process prints its rank, the total number of processes and the value he received from the process 0.

In your home directory, create a file `mpi_broadcast.c` and copy the following source code:


        #include <stdio.h>
        #include <mpi.h>
        #include <unistd.h>
        #include <time.h> /* for the work function only */

        int main (int argc, char *argv []) {
               char hostname[257];
               int size, rank;
               int i, pid;
               int bcast_value = 1;

               gethostname(hostname, sizeof hostname);
               MPI_Init(&argc, &argv);
               MPI_Comm_rank(MPI_COMM_WORLD, &rank);
               MPI_Comm_size(MPI_COMM_WORLD, &size);
               if (!rank) {
                    bcast_value = 42;
               }
               MPI_Bcast(&bcast_value,1 ,MPI_INT, 0, MPI_COMM_WORLD );
               printf("%s\t- %d - %d - %d\n", hostname, rank, size, bcast_value);
               fflush(stdout);

               MPI_Barrier(MPI_COMM_WORLD);
               MPI_Finalize();
               return 0;
        }

Reserve 2 cores on two distinct node with OAR

        (access-gaia)$> oarsub -I -l nodes=2/core=1

or with Slurm

        (access-iris)$> srun -p interactive --qos qos-interactive --time 1:00:0 -N 2 -n 2 --pty bash


Load a toolchain and compile the code using `mpicc`

        (node)$> mpicc mpi_broadcast.c -o mpi_broadcast -lpthread

If you use OAR, execute your mpi program using `mpirun`.
Note that the `-n` parameter of mpirun is the number of processes, which should be equal to the number of reserved cpu cores most of the time.

        (node)$> OAR_NTASKS=$(cat $OAR_NODEFILE | wc)
        (node)$> mpirun -n $OAR_NTASKS -hostfile $OAR_NODEFILE ~/mpi_broadcast

If you use Slurm, you can use the `srun` command. Create an interactive job, with 2 nodes (`-N 2`), and at least 2 tasks (`-n 2`).

        (node)$> srun -n $SLURM_NTASKS ~/mpi_broadcast


### Using SSH proxycommand setup to access the clusters despite port filtering

It might happen that the port 8022 is filtered from your working place. You can easily bypass this firewall rule using an SSH proxycommand to setup transparently multi-hop connexions *through* one host (a gateway) to get to the access frontend of the cluster, as depited below:

    [laptop] -----||--------> 22 [SSH gateway] ---------> 8022 [access-{chaos,gaia}]
               firewall

The gateway can be any SSH server which have access to the access frontend of the cluster. The [Gforge @ UL](http://gforge.uni.lu) is typically used in this context but you can prefer any other alternative (your personal NAS @ home etc.). Then alter the SSH config on your laptop (in `~/.ssh/config` typically) as follows:

* create an entry to be able to connect to the gateway:

#### Alias for the gateway (not really needed, but convenient), below instantiated

    Host gw
    User anotherlogin
    Hostname host.domain.org
    ForwardAgent no

#### Automatic connection to UL HPC from the outside via the gateway

    Host *.ulhpc
    ProxyCommand ssh gw "nc -q 0 `basename %h .ulhpc` %p"

Ensure you can connect to the gateway:

    (laptop)$> ssh gw
    (gateway)$> exit # or CTRL-D

The `.ulhpc` suffix we mentioned in the previous configuration is an arbitrary suffix you will now specify in your command lines in order to access the UL HPC platform via the gateway as follows:

    (laptop)$> ssh gaia.ulhpc
