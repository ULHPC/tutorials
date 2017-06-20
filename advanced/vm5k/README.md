-*- mode: markdown; mode: visual-line; fill-column: 80 -*-

Copyright (c) 2015 Hyacinthe Cartiaux <Hyacinthe.Cartiaux@uni.lu><hpc-sysadmins@uni.lu>

---------------------------------------------------------------
# Deploying virtual machines with Vm5k on Grid'5000

[![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/advanced/vm5k/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/advanced/vm5k/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

Grid’5000 is a scientific instrument distributed in 10 sites (mainly in France)
for research in large-scale parallel and distributed systems. It aims at providing
a highly reconfigurable, controllable and monitorable experimental platform to its users.

The infrastructure has reached 1035 nodes and 7782 cores and all sites are connected
to RENATER with a 10 Gb/s link, except Reims and Nantes (1 Gb/s).

![G5K map](https://www.grid5000.fr/mediawiki/images/Renater5-g5k.jpg)

The objectives of this tutorial are:

* connect to Grid'5000
* discover the basic features of Grid'5000
* use vm5k in order to deploy virtual machines on the grid

# Getting started

## User charter

You should first read the [User Charter](https://www.grid5000.fr/mediawiki/index.php/Grid5000:UserCharter).
The mains points are:

* maintain your [user reports](https://api.grid5000.fr/sid/reports/_admin/index.html) up-to-date (publications, experiments, etc)
* during working days and office hours (09:00 to 19:00), you should not use more than the equivalent of 2 hours of all the resources available in a cluster
* your jobs should not cross the 09:00 and 19:00 boundaries during week days
* you should not have more than 2 reservations in advance

Basically: develop your experiments during day time, launch them over the nights and week-ends

## Account

Fill the [account request form](https://www.grid5000.fr/mediawiki/index.php/Special:G5KRequestAccountUMS)
At the Manager entry where you’re asked the Grid’5000 login of the person
who’s responsible of the account you’re requesting, answer **svarrett**

## Connection

Grid'5000 provided 2 national access servers, located in Lille and Sophia.
These servers allow the user to connect the site's frontend.

    (user)   ssh <login>@access.grid5000.fr
    (access) ssh luxembourg

As an alternative, we can connect directly to the Luxembourg frontend from within the UL network:

    (user)   ssh <login>@grid5000.uni.lu


## Reservation and deployment

Grid'5000 uses OAR, all the oar commands you use on Gaia and Chaos clusters are valid.

* [OAR documentation on hpc.uni.lu](https://hpc.uni.lu/users/docs/oar.html)

Additionally to the computing nodes, G5K provides more resources:

* subnets (ranges of IP for virtualization / cloud experiments)
* vlan (reconfigure the network equipments)
* storage (iscsi / nfs)
* ...

The job type "deploy" is also supported, which means that you can use kadeploy to reinstall
a cluster node and gain root access during the time of your jobs

## Tutorials

It is highly recommended to read and follow the [Getting Started tutorial](https://www.grid5000.fr/mediawiki/index.php/Getting_Started),
and all the others tutorials available in the [User Portal](https://www.grid5000.fr/mediawiki/index.php/Category:Portal:User)

# VM5K

[Vm5k](https://vm5k.readthedocs.org/) is a tool used to deploy a large number of virtual machines on the Grid‘5000 platform.

In short, Vm5k

* manages the reservation, locally if you work on one site, or globally with `oargridsub`
* install the hosts with kadeploy, and configure the virtualization stack for you
* configure the network (bridges)
* deploy the virtual machines


## Installation (from git)

You will install VM5K in your home directory.

* Specify the proxy configuration

        (frontend) export http_proxy="http://proxy:3128"
        (frontend) export https_proxy="https://proxy:3128"

* Install execo, which is a dependency of vm5k

        (frontend) easy_install --user execo

* Clone the git repository of vm5k and install it

        (frontend) git clone https://github.com/lpouillo/vm5k.git
        (frontend) cd vm5k
        (frontend) python setup.py  install --user

* In your bashrc file, add the ~/.local/bin directory to the PATH environment variable

        (frontend) echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc

## Usage

## Basic features

Each deployment takes around 20 minutes, so you don't have to execute all the following examples:

* spawn 20 VMs on the granduc cluster, with a walltime of 30 minutes, and write the output files in the directory `vm5k_test`:

        (frontend) vm5k --n_vm 10 -r granduc -w 0:30:00 -o vm5k_test

You'll find the list of VMs with their ips in the file vm5k_test/vms.list

    10.172.1.45     vm-1
    10.172.1.46     vm-2
    10.172.1.47     vm-3
    10.172.1.48     vm-4
    10.172.1.49     vm-5
    10.172.1.50     vm-6
    10.172.1.51     vm-7
    10.172.1.52     vm-8
    ...

* Let's spawn 100 VMs on 2 hosts in Nancy and Luxembourg

        (frontend) vm5k --n_vm 10 -r nancy:1,luxembourg:1 -w 0:30:00 -o vm5k_test

* We can also specify the VM template (resources) with the parameter `--vm_template`

        (frontend) vm5k --n_vm 10 -r nancy:2,luxembourg:2 -w 0:30:00 -o vm5k_test --vm_template '<vm mem="4096" hdd="10" n_cpu="4" cpuset="auto"/>'

### Distribution

* Balance the nodes on all the reserved hosts

        (frontend) vm5k --n_vm 100 -r granduc:4 -o vm5k_test -d n_by_hosts

* Concentrate the VMs on a minimal number of hosts

        (frontend) vm5k -r grid5000:20 -n 100 -o vm5k_test -d concentrated

### Advanced feature: define the deployment topology

You can control the deployment topology, and specify finely the clusters, nodes and virtual machines per node.

Create a file named `topology.xml`, and change the sites, cluster and host id as needed:

    <vm5k>
      <site id="luxembourg">
        <cluster id="granduc">
          <host id="granduc-2">
            <vm mem="2048" hdd="4" id="vm-33" cpu="1"/>
            <vm mem="2048" hdd="4" id="vm-34" cpu="1"/>
            <vm mem="2048" hdd="4" id="vm-35" cpu="1"/>
          </host>
          <host id="granduc-3">
            <vm mem="2048" hdd="4" id="vm-43" cpu="1"/>
            <vm mem="2048" hdd="4" id="vm-44" cpu="1"/>
            <vm mem="2048" hdd="4" id="vm-45" cpu="1"/>
            <vm mem="2048" hdd="4" id="vm-43" cpu="1"/>
            <vm mem="2048" hdd="4" id="vm-44" cpu="1"/>
          </host>
        </cluster>
      </site>
      <site id="nancy">
        <cluster id="graphene">
          <host id="graphene-30">
            <vm mem="2048" hdd="4" id="vm-30" cpu="1"/>
            <vm mem="2048" hdd="4" id="vm-31" cpu="1"/>
          </host>
        </cluster>
      </site>
    </vm5k>


Give this file to vm5k

    (frontend) vm5k -i topology.xml -w 0:30:0 -o vm5k_test


## Experiment

We will deploy 100 VMs and deploy the munin monitoring software.
This is an example, munin will allow you to monitor the activity on all the VMs.

### Install munin clients on all the other servers

* Spawn 100 VMs

        (frontend) vm5k --n_vm 50 -w 2:00:00 -r luxembourg -o hpcschool2015 -o vm5k_xp

* Launch a script on all the VMs after their deployment, we will use `taktuk` (you could also clush, pdsh, etc)

        (frontend) taktuk -l root -f vm5k_xp/vms.list broadcast exec [ apt-get update ]
        (frontend) taktuk -l root -f vm5k_xp/vms.list broadcast exec [ apt-get install -y munin-node stress ]
        (frontend) taktuk -l root -f vm5k_xp/vms.list broadcast exec [ 'echo cidr_allow 10.0.0.0/8 >> /etc/munin/munin-node.conf' ]
        (frontend) taktuk -l root -f vm5k_xp/vms.list broadcast exec [ '/etc/init.d/munin-node restart' ]

### Install munin server on the first physical host

* Choose the first virtual machines

        (frontend) head -n 1 vm5k_xp/vms.list
        10.172.1.45     vm-1

* Transfer the list of virtual machines to the VM

        (frontend) scp vm5k_xp/vms.list root@10.172.1.45:/tmp/

* Connect to the VM in order to install and configure munin

        (frontend) ssh root@10.172.1.45

        (vm-1) apt-get install munin apache2

* Configure the Apache http server

        (vm-1) sed -i '/[aA]llow/d' /etc/apache2/conf.d/munin
        (vm-1) apache2ctl restart

* Generate the munin configuration

        (vm-1) cat /tmp/vms.list  | awk '{print "["$2".g5k]\n    address "$1"\n    use_node_name yes\n"}' >> /etc/munin/munin.conf
        (vm-1) /etc/init.d/munin restart


### Connect to munin

* Let's generate a fake activity, stress the VM during 60 seconds

        (frontend) taktuk -l root -f hpcschool2015/vms.list broadcast exec [ 'stress -c 1 -t 60' ]

* Open a ssh tunnel on port 80

        (user) ssh -L1080:10.172.1.45:80 <login>@grid5000.uni.lu

Open a browser and navigates to <http://localhost:1080>
