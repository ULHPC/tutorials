-*- mode: markdown; mode: auto-fill; fill-column: 80 -*-
`README.md`

Copyright (c) 2014 [Sebastien Varrette](mailto:<Sebastien.Varrette@uni.lu>) [www](http://varrette.gforge.uni.lu)

        Time-stamp: <Fri 2014-02-28 10:41 svarrette>

-------------------

# UL HPC Tutorial: Getting Started

This tutorial will guide you through your first steps on the
[UL HPC platform](http://hpc.uni.lu).  

Before proceeding: 

* make sure you have an account (if not, follow [this procedure](https://hpc.uni.lu/get_an_account)), and an SSH client.
* take a look at the [quickstart guide](https://hpc.uni.lu/users/getting_started.html)
* ensure you operate from a Linux / Mac environement. Most commands below assumes running in a Terminal in this context. If you're running Windows, you can use Putty tools etc. as described [on this page](https://hpc.uni.lu/users/docs/access.html#installing-ssh-on-windows) yet it's probably better that you familiarize "natively" with Linux-based environment by having a Linux Virtual Machine (consider for that [VirtualBox](https://www.virtualbox.org/)). 

From a general perspective, the [Support page](https://hpc.uni.lu/users/docs/report_pbs.html) describes how to get help during your UL HPC usage. 

## Convention

In the below tutorial, you'll proposed terminal commands where the prompt is denoted by `$>`. 

In general, we will prefix to precise the execution context (_i.e._ your laptop, a cluster frontend or a node). Remember that `#` character is a comment. Example: 

		# This is a comment 
		$> hostname

		(laptop)$> hostname         # executed from your personal laptop / workstation

		(access-gaia)$> hostname    # executed from access server of the Gaia cluster
 

## Platform overview. 

You can find a brief overview of the platform with key characterization numbers [on this page](https://hpc.uni.lu/systems/overview.html).

The general organization of each cluster is depicted below:

![UL HPC clusters general organization](https://hpc.uni.lu/images/overview/clusters_general_organization.png)

Details on this organization can be found [here](https://hpc.uni.lu/systems/clusters.html#clusters-organization)


# Connecting for the first time and preparing your SSH environment

* [Access / SSH Tutorial](https://hpc.uni.lu/users/docs/access.html)

The way SSH handles the keys and the configuration files is illustrated in the following figure:

![SSH key management](https://hpc.uni.lu/images/docssh/schema.png)

In order to be able to login to the clusters, you have sent us through the Account request form the **public key** (i.e. `id_dsa.pub`, `id_rsa.pub` or the **public key** as saved by PuttY) you initially generated, enabling us to configure the `~/.ssh/authorized_keys` file of your account.  


### Step 1: Connect to UL HPC 

Run the following commands in a terminal (substituting *yourlogin* with the login name you received from us):

        (laptop)$> ssh -p 8022 yourlogin@access-chaos.uni.lu

If you want to connect to the gaia cluster, 

        (laptop)$> ssh -p 8022 yourlogin@access-gaia.uni.lu

Now you probably want to avoid taping this long command to connect to the platform. You can customize SSH aliases for that. Edit the file `~/.ssh/config` (create it if it does not already exist) and adding the following entries: 

        Host chaos-cluster
            Hostname access-chaos.uni.lu

        Host gaia-cluster
            Hostname access-gaia.uni.lu

        Host *-cluster
            User yourlogin
            Port 8022
            ForwardAgent no

Now you shall be able to issue the following (simpler) command to connect to the cluster and obtain the welcome banner: 

		(laptop)$> ssh gaia-cluster

		(laptop)$> ssh chaos-cluster

In the sequel, we assume these aliases to be defined. 
		
## Step 2: configure your SSH environment on all clusters

The SSH key you provided us secure your connection __from__ your laptop (or personal workstation) __to__ the cluster frontends. It is thus important to protect them by a passphrase. 

We will now configure a passphrase-free SSH key pair to permit a transparent connection from one cluster to another. 


* Connect to the `chaos` cluster: 

		(laptop)$> ssh chaos-cluster
		
* generate a new SSH key pair with `ssh-keygen` (leave the passphrase empty):

		(access-chaos)$> ssh-keygen -t dsa
		Generating public/private dsa key pair.
		Enter file in which to save the key (/home/users/yourlogin/.ssh/id_dsa): 
		Enter passphrase (empty for no passphrase): 
		Your identification has been saved in /home/users/yourlogin/.ssh/id_dsa.
		Your public key has been saved in /home/users/yourlogin/.ssh/id_dsa.pub.
		The key fingerprint is:
		1f:1d:a5:66:3b:4a:68:bc:7d:8c:7f:33:c9:77:0d:4a yourlogin@access.chaos-cluster.uni.lux
		The key's randomart image is:
		`-[ DSA 1024]---`
		|              .  |
		|             o   |
		|            =    |
		|       . . + o   |
		|        S o +    |
		|       . = =E..  |
		|        . =.oo o.|
		|           o. * +|
		|            .. +.|
		`---------------`

* authorize the newly generated public key to be used during challenge/response authentication:

		(access-chaos)$> cat ~/.ssh/id_dsa.pub 
		ssh-dss AAAAB[...]B2== yourlogin@access.chaos-cluster.uni.lux
		(access-chaos)$> cat ~/.ssh/id_dsa.pub  >> ~/.ssh/authorized_keys
		
  * you can check that it works by connecting to localhost: 
  
  		(access-chaos)$> ssh -p 8022 localhost
  		[...]
  		(access-chaos)$> exit   # or CTRL-D
  		
* add an alias to facilitate the connection to each cluster by adding the following SSH configuration entry in the file `~/.ssh/config`: 

		Host gaia chaos
    		User yourlogin
    		Port 8022

		Host gaia
    		Hostname access-gaia.uni.lu
		Host chaos
    		Hostname access-chaos.uni.lu
  		
You'll have to setup the same key package on the gaia cluster such that you can then work indefferently on one or another cluster. It's also the occasion to learn how to add a new SSH key to your authorized key portfolio. 

* Open another terminal and connect to the gaia cluster 

		(laptop)$> ssh gaia-cluster

* edit the file `~/.ssh/authorized_keys` to add your previously generated key on chaos (use `:wq` in [vim](http://tnerual.eriogerg.free.fr/vimqrc.pdf) to save and quit):

		(access-gaia)$> vim ~/.ssh/authorized_keys


* go back to the terminal where you're connected on chaos, you shall now be able to connect to gaia, and reversely: 

		(access-chaos)$> ssh gaia
		[...]
		(access-gaia)$> exit     # or CRTL-D
		
You have a different home directory on each UL HPC site, so you will usually use Rsync or scp to move data around (see [transfering files tutorials](https://hpc.uni.lu/users/docs/filetransfer.html)).

Now that we are able to connect __from__ chaos __to__ gaia, we will transfer the SSH keys and configuration in place from chaos and check that we can connnect back: 

		(access-chaos)$> scp ~/.ssh/id_dsa* gaia:.ssh/
		(access-chaos)$> scp ~/.ssh/config  gaia:.ssh/
		(access-chaos)$> ssh gaia
		[...]
		(access-gaia)$>  ssh chaos
		(access-chaos)$> exit     # or CRTL-D
		(access-gaia)$>  exit     # or CRTL-D

So now **we have setup a bi-directional transparent connection from one cluster to the other.** 

## Step 2bis: Using SSH proxycommand setup to access the clusters despite port filtering

It might happen that the port 8022 is filtered from your working place. You can easily bypass this firewall rule using an SSH proxycommand to setup transparently multi-hop connexions *through* one host (a gateway) to get to the access frontend of the cluster, as depited below: 

     [laptop] -----||--------> 22 [SSH gateway] ---------> 8022 [access-{chaos,gaia}]
                firewall

The gateway can be any SSH server which have access to the access frontend of the cluster. The [Gforge @ UL](http://gforge.uni.lu) is typically used in this context but you can prefer any other alternative (your personal NAS @ home etc.). Then alter the SSH config on yout laptop (in `~/.ssh/confg` typically) as follows:

* create an entry to be able to connect to the gateway:

		# Alias for the gateway (not really needed, but convenient), below instanciated 
		Host gw
		    User anotherlogin
		    Hostname host.domain.org
		    ForwardAgent no

		# Automatic connection to UL HPC from the outside via the gateway
		Host *.ulhpc
			ProxyCommand ssh gw "nc -q 0 `basename %h .ulhpc` %p"

* ensure you can connect to the gateway:

		(laptop)$> ssh gw
		(gateway)$> exit # or CTRL-D 
		
* the `.ulhpc` suffix we mentionned in the previous configuration is an arbitrary suffix you will now specify in your command lines in order to access the UL HPC platform via the gateway as follows: 

		(laptop)$> ssh gaia.ulhpc

# Discovering, visualizing and reserving UL HPC resources

## Step 1: the working environment 

* [reference documentation](http://ulhpc_www.dev/users/docs/env.html)

After a successful login onto one of the access node (see [Cluster Access](https://hpc.uni.lu/users/docs/access.html)), you end into your personal homedir `$HOME` which is shared over NFS between the access node and the computing nodes.

Again, remember that your homedir is placed on __separate__ NFS servers on each site, which __ARE NOT SYNCHRONIZED__: data synchronization between each of them remain at your own responsability. We will see below that the UL HPC team prepared for you a script to facilitate the transfer of data between each site. 

Otherwise, you have to be aware of at least two directories: 

* `$HOME`: your home directory under NFS. 
* `$SCRATCH`: a non-backed up area put if possible under Lustre for fast I/O operations

Your homedir is under a regular backup policy. Therefore you are asked to pay attention to your disk usage __and__ the number of files you store there. 

* estimate file space usage and summarize disk usage of each FILE, recursively for directories using the `ncdu` command:

		(access)$> ncdu
		
* you shall also pay attention to the number of files in your homedirectory. You can count them as follows: 

		(access)$> find . -type f | wc -l

## Step 2: web monitoring interfaces 

Each cluster offers a set of web services to monitore the platform usage: 

* A [pie-chart overview of the platform usage](https://hpc.uni.lu/status/overview.html)
* [Monika](https://hpc.uni.lu/status/monika.html), the visualization interface of the OAR scheduler, which  display the status of the clusters as regards the jobs running on the platform.
* [DrawGantt](https://hpc.uni.lu/status/drawgantt.html), the Gantt visualization of jobs scheduled on OAR
* [Ganglia](https://hpc.uni.lu/status/ganglia.html), a scalable distributed monitoring system for high-performance computing systems such as clusters and Grids.
* [CDash](http://cdash.uni.lux/) (internal UL network use)

## Step 3: Reserving resources with OAR: the basics

* [reference documentation](https://hpc.uni.lu/users/docs/oar.html)

[OAR](http://oar.imag.fr/) is an open-source batch scheduler which provides simple yet flexible facilities for the exploitation of the UL HPC clusters.

* Connect to one of the UL HPC  frontend. You can request resources in interactive mode:

		(access)$> oarsub -I 

  Notice that with no parameters, oarsub gave you one resource (one core) for two hour. You were also directly connected to the node you reserved with an interactive shell.
  No exit the reservation: 
  
        (node)$> exit      # or CTRL-D
  
  When you run exit, you are disconnected and your reservation is terminated. 
  
To avoid anticipated termination of your jobs in case or errors (terminal closed by mistake), 
you can reserve and connect in 2 steps using the job id associated to your reservation. 

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

## Step 4: Job management

Normally, the previously run job is still running.

* You can check the status of your running jobs using `oarstat` command:

		(access)$> oarstat      # access all jobs 
		(access)$> oarstat -u   # access all your jobs
		
Then 
























			
	



 




