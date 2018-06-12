-*- mode: markdown; mode: visual-line; fill-column: 80 -*-

Copyright (c) 2014-2017 UL HPC Team  <hpc-sysadmins@uni.lu>

---------------------------------------------------------------
# UL HPC Tutorial: Create and reproduce work environments using Vagrant

[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/advanced/Vagrant) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/advanced/Vagrant/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)


`/!\ IMPORTANT` Up-to-date instructions for Vagrant can be found in the ["Reproducible Research at the Cloud Era"](http://rr-tutorials.readthedocs.io/) Tutorial. Below instructions are probably outdated but kept for archive purposes.

---------------

Vagrant is a tool that allows to easily and rapidly create and configure reproducible and portable work environments using Virtual Machines. This is especially useful if you want to test your work in a stable and controlled environment and minimize the various unintended or untrackable changes that may occur on a physical machine.

In this tutorial, we are going to explain the steps to install Vagrant and create your first basic Linux Virtual Machine with it.

## Vagrant installation

**Prerequisite:**

Vagrant can use many Virtual Machine providers such as [VirtualBox](https://www.virtualbox.org/), [VMware](http://www.vmware.com/) and [Docker](https://www.docker.com/) with VirtualBox being the easiest to use, and the default option in Vagrant.

Our first step is to install VirtualBox, you can download and install the correct version for your operating system from the [official website](https://www.virtualbox.org/wiki/Downloads). In many Linux distributions it is provided as a package from the standard repositories, thus you can use your usual package manager to install it.

Once this prerequisite is met, we can install Vagrant. Download the correct version for your operating system on the [official website](http://www.vagrantup.com/downloads) and install it.

## Using Vagrant to create a Virtual Machine

The main advantage of Vagrant is that it lets you import and use pre-configured Virtual Machines (called `boxes` in this context) which can become bases for your own customizations (installed applications, libraries, etc). With Vagrant it becomes really fast and effortless to create and run a new Virtual Machine.

The Vagrant boxes contain the disk image of a VM without the virtual hardware details of the VM, which are initialized by Vagrant and can be edited by the user.

The first step is to choose a pre-configured box to use. It is possible to create your own from scratch yet this is not in the scope of the current tutorial.
Freely available boxes can be found at the following two main sources:

- [Atlas corps box catalog](https://atlas.hashicorp.com/boxes/search)
- [vagrantbox.es catalaog](http://www.vagrantbox.es/)

The first catalog is the default box download location for Vagrant. This means that you can directly use the name of the boxes you find here with Vagrant (e.g. `ubuntu/trusty64`).
To use the second catalog you would additionaly need to provide the source box URL, yet this catalog provides a much richer variety of boxes.

### Adding a new box

To add a box and make it usable in Vagrant, we are going to use the `vagrant box add` command. In the example below we will add one box from each of the catalogs in order to present the different possibilities.
We are going to add the `ubuntu/trusty64` box from the Atlas catalog and the `Ubuntu 14.04` box (by its [url](https://github.com/kraksoft/vagrant-box-ubuntu/releases/download/14.04/ubuntu-14.04-amd64.box)) from the vagrantbox.es catalog.

To add the first box, we use the following command (which may take some time due to the time needed to download the box):

        $> vagrant box add ubuntu/trusty64
        ==> box: Loading metadata for box 'ubuntu/trusty64'
            box: URL: https://vagrantcloud.com/ubuntu/trusty64
        ==> box: Adding box 'ubuntu/trusty64' (v14.04) for provider: virtualbox
            box: Downloading: https://vagrantcloud.com/ubuntu/boxes/trusty64/versions/14.04/providers/virtualbox.box
        ==> box: Successfully added box 'ubuntu/trusty64' (v14.04) for 'virtualbox'!
In this case, you just had to give the name of the box and Vagrant found the box by itself and added the box under the `ubuntu/trusty64` name.

To list the local boxes available to Vagrant for initialization of new VMs, we use the `vagrant box list` command:

        $> vagrant box list
        ubuntu/trusty64    (virtualbox, 14.04)

To add the second box, you need to use a slightly different syntax since you need to precise the name you want to give to the box as well as its source URL:

        $> vagrant box add ubuntu14.04 https://github.com/kraksoft/vagrant-box-ubuntu/releases/download/14.04/ubuntu-14.04-amd64.box
        ==> box: Adding box 'ubuntu14.04' (v0) for provider:
            box: Downloading: https://github.com/kraksoft/vagrant-box-ubuntu/releases/download/14.04/ubuntu-14.04-amd64.box
        ==> box: Successfully added box 'ubuntu14.04' (v0) for 'virtualbox'!
Now a second box will be available to Vagrant under the name `ubuntu14.04`:

        $> vagrant box list
        ubuntu/trusty64    (virtualbox, 14.04)
        ubuntu14.04        (virtualbox, 0)

In the rest of the tutorial we are only going to use the first box. To remove a box we use the `vagrant box remove` command as follows:

        $> vagrant box remove ubuntu14.04
        Removing box 'ubuntu14.04' (v0) with provider 'virtualbox'...
Checking that it has been removed:

        $> vagran box list
        ubuntu/trusty64    (virtualbox, 14.04)

### Creating a new Virtual Machine

Now we are going to create a new Virtual Machine using the `ubuntu/trusty64` box.
We will initialize it in an empty directory (which is not absolutely mandatory):

        $> mkdir vagrant && cd vagrant

Next, we make Vagrant prepare the configuration file describing the VM:

        $> vagrant init ubuntu/trusty64
        A `Vagrantfile` has been placed in this directory. You are now
        ready to `vagrant up` your first virtual environment! Please read
        the comments in the Vagrantfile as well as documentation on
        `vagrantup.com` for more information on using Vagrant.
You should now see a file named `Vagrantfile` in your directory. This file contains the minimal information for Vagrant to launch the VM. We could modify it to set up specific parameters of the VM (number of virtual cores, memory size, etc), but this constitutes advanced usage for which full documentation that can be found on the [official site](http://docs.vagrantup.com/v2/). However, it may be interesting to understand what is actually needed in this file, since it contains a lot of commented information.
The minimal content of a `Vagrantfile` is as follows:

        VAGRANTFILE_API_VERSION = "2"
        Vagrant.configure("VAGRANTFILE_API_VERSION") do |config|
            config.vm.box = "hashicorp/trusty64"
        end
This basically defines which version of the Vagrant API will be used to build the VM using the box given as a base.

Now, to launch the VM you only need to use the single `vagrant up` command in the same directory where the `Vagrantfile` exists (this may take some time since Vagrant is going to boot the VM and set its basic configuration):

        $> vagrant up
        Bringing machine 'default' up with 'virtualbox' provider...
        ==> default: Importing base box 'ubuntu/trusty64'...
        ==> default: Matching MAC address for NAT networking...
        ==> default: Checking if box 'ubuntu/trusty64' is up to date...
        ==> default: Setting the name of the VM: vagrant_default_1425476252413_67101
        ==> default: Clearing any previously set forwarded ports...
        ==> default: Clearing any previously set network interfaces...
        ==> default: Preparing network interfaces based on configuration...
            default: Adapter 1: nat
        ==> default: Forwarding ports...
            default: 22 => 2222 (adapter 1)
        ==> default: Booting VM...
        ==> default: Waiting for machine to boot. This may take a few minutes...
            default: SSH address: 127.0.0.1:2222
            default: SSH username: vagrant
            default: SSH auth method: private key
            default: Warning: Connection timeout. Retrying...
            default: Warning: Remote connection disconnect. Retrying...
        ==> default: Machine booted and ready!
        ==> default: Checking for guest additions in VM...
        ==> default: Mounting shared folders...
            default: /vagrant => /tmp/vagrant
Your VM is now up and running at this point. To access it, use the `vagrant ssh` command within the same directory :

        $> vagrant ssh
You should now be connected to your VM and ready to work.

An interesting feature of Vagrant is that your computer (the "host") shares the directory that contains the `Vagrantfile` with your VM (the "guest"), where it is seen as `/vagrant`.

Assuming you have a script or data files you want to access from within the VM, you simply put them in the same directory as the `Vagrantfile` and then use them in the VM under `/vagrant`. The reverse is also true.

To learn more than the basics covered in this tutorial, we encourage you to refer to the [official documentation](http://docs.vagrantup.com/v2/).
