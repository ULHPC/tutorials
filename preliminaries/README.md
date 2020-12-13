[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/preliminaries/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/preliminaries/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/preliminaries/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)


---------------------------------------
# Preliminaries to ULHPC facility access

     Copyright (c) 2020 UL HPC Team <hpc-team@uni.lu>

<!-- [![](https://github.com/ULHPC/tutorials/raw/devel/preliminaries/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/preliminaries/slides.pdf) -->

Welcome to the High Performance Computing (HPC) Facility of the University of Luxembourg (ULHPC)!

To take the best out of the different tutorials and practical sessions proposed in the training sessions organised by the University of Luxembourg, you have to follow several steps to configure your working environments.
In particular, ensure you have followed the [preliminary setup instructions](setup/preliminaries.md) for your laptop.

**Convention**

In the below tutorial, you'll proposed terminal commands where the prompt is denoted by `$>`.

In general, we will prefix to precise the execution context (_i.e._ your laptop, a cluster frontend or a node). Remember that `#` character is a comment. Example:

                # This is a comment
                $> hostname

                (laptop)$> hostname         # executed from your personal laptop / workstation

                (access-iris)$> hostname    # executed from access server of the Iris cluster


## Git Installation

```bash
# Mac OS X, using Homebrew - https://brew.sh
brew install git git-gui git-flow gitk tig kdiff3

# Ubuntu / WSL / RedHat
{ sudo apt | dnf } install git-core git-flow tig gitk kdiff3

# Windows, using Chocolatey - https://chocolatey.org/
$> choco.exe install git gitflow-avh
```

### Git Initial Setup

```bash
# /!\ To add to your bash/zsh profile (~/.profile)
# XDG Base Directory Specification
# See https://specifications.freedesktop.org/basedir-spec/latest/
(laptop)$> export XDG_CONFIG_HOME=$HOME/.config
(laptop)$> export XDG_CACHE_HOME=$HOME/.cache
(laptop)$> export XDG_DATA_HOME=$HOME/.local/share

# Create the directories
mkdir -p ~/.config/git $XDG_CACHE_HOME $XDG_DATA_HOME

# Basic Git defaults
(laptop)$> git config --global user.name "Firstname LastName"
(laptop)$> git config --global user.email "<email>@<domain>"
(laptop)$> git config --global user.signingkey <gpg-keyID>     # <-- Leave this part if you don't have a gpg-keyID
(laptop)$> git config --global color.ui true

# Set your default editor -- vim in this case
(laptop)$> git config ---global core.editor vim
```

---------------------
## Secure SHell (SSH)


* [Access / SSH Tutorial](https://hpc.uni.lu/users/docs/access.html)

The way SSH handles the keys and the configuration files is illustrated in the following figure:

![SSH key management](https://hpc.uni.lu/images/docssh/schema.png)


Developed by [SSH Communications Security Ltd.](http://www.ssh.com), Secure Shell is a program to log into another computer over a network, to execute commands in a remote machine, and to move files from one machine to another in a secure way.

It provides strong authentication and secure communications over insecure channels. To use SSH, you have to generate a pair of keys, one **public** and the other **private**.
The public key authentication is the most secure and flexible approach to ensure a multi-purpose transparent connection to a remote server. You will learn here how to generate an SSH key pair, authorize its **public** part on the ULHPC portal, allowing to connect and transfer data securely data toward the ULHPC facility.


### SSH Key Generation

(_Again_) ensure your have followed the [preliminary setup instructions](setup/preliminaries.md) as the below guideline is common to **ALL** platforms (including Windows assuming you have configure Ubuntu over WSL)

Open a Terminal.
SSH is installed natively on your machine and the `ssh` command should be accessible from the command line:

```bash
(laptop)$> ssh -V
OpenSSH_7.9p1, LibreSSL 2.7.3
```

#### SSH Key Management

You can check all available SSH keys on your computer by running the following command on your terminal:

```bash
(laptop)$> for key in ~/.ssh/id_*; do ssh-keygen -l -f "${key}"; done | uniq
```

Your SSH keys might use one of the following algorithms:

* _DSA_: It's unsafe and even no longer supported since OpenSSH version 7, you need to upgrade it!
* _RSA_: OK if the key size has 3072 or 4096-bit length -- 1024-bit length is considered unsafe.
* _Ed25519_: It’s the most recommended public-key algorithm available today

#### Default RSA Key Pair

To generate an RSA SSH keys **of 4096-bit length**, just use the `ssh-keygen` command as follows:

```bash
(laptop)$> ssh-keygen -t rsa -b 4096 -o -a 100
Generating public/private rsa key pair.
Enter file in which to save the key (/home/user/.ssh/id_rsa):
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /home/user/.ssh/id_rsa.
Your public key has been saved in /home/user/.ssh/id_rsa.pub.
The key fingerprint is:
fe:e8:26:df:38:49:3a:99:d7:85:4e:c3:85:c8:24:5b username@yourworkstation
The key's randomart image is:
+---[RSA 4096]----+
|                 |
|      . E        |
|       * . .     |
|      . o . .    |
|        S. o     |
|       .. = .    |
|       =.= o     |
|      * ==o      |
|       B=.o      |
+-----------------+
```

**IMPORTANT: To ensure the security of your key-pair, you MUST protect your SSH keys with a passphrase!**

After the execution of `ssh-keygen` command, the keys are generated and stored in the following files:

* SSH RSA Private key: `~/.ssh/id_rsa`. Again, **NEVER EVER TRANSMIT THIS FILE**
* SSH RSA Public key:  `~/.ssh/id_rsa.pub`.  **This file is the ONLY one SAFE to distribute**

Ensure the access rights are correct on the generated keys using the ' `ls -l` ' command. The private key should be readable only by you:

```bash
(laptop)$> ls -l ~/.ssh/id_*
-rw------- 1 username groupname 751 Mar  1 20:16 /home/username/.ssh/id_rsa
-rw-r--r-- 1 username groupname 603 Mar  1 20:16 /home/username/.ssh/id_rsa.pub
```

#### ED25519 Key Pair

**ED25519 is the most recommended public-key algorithm available today.**

Repeat the procedure to generate a

```bash
(laptop)$> ssh-keygen -t ed25519 -o -a 100
[...]
$> ls -l ~/.ssh/id_*
-rw------- 1 username groupname 751 Mar  1 20:16 /home/username/.ssh/id_rsa
-rw-r--r-- 1 username groupname 603 Mar  1 20:16 /home/username/.ssh/id_rsa.pub     # Public  RSA key
-rw------- 1 username groupname 751 Mar  1 20:16 /home/username/.ssh/id_ed25519
-rw-r--r-- 1 username groupname 603 Mar  1 20:16 /home/username/.ssh/id_ed25519.pub # Public ED25519 key
```

### Upload the keys on the ULHPC Identity Management Portal

You should now upload your public SSH keys `*.pub` to your user entry on the  ULHPC Identity Management Portal.

Connect to the IdM portal and enter your **ULHPC** credentials.

![](https://access.redhat.com/webassets/avalon/d/Red_Hat_Enterprise_Linux-7-Linux_Domain_Identity_Authentication_and_Policy_Guide-en-US/images/0b67bd0c53b2b26b1d9ce416280f1e83/web_ui_login_screen.png)

First copy the content of the key you want to add

``` bash
# Example with ED25519 **public** key
(laptop)$> cat ~/.ssh/id_ed25519.pub
ssh-ed25519 AAAA[...]
# OR the RSA **public** key
(laptop)$> cat ~/.ssh/id_rsa.pub
ssh-rsa AAAA[...]
```

Then on the portal:

1. Select Identity / Users.
2. Select your login entry
3. Under the Settings tab in the Account Settings area, click SSH public keys: **Add**.

![](https://access.redhat.com/webassets/avalon/d/Red_Hat_Enterprise_Linux-7-Linux_Domain_Identity_Authentication_and_Policy_Guide-en-US/images/162d5680e990e7cb5f2629377a5d288a/sshkeys-user1.png)

Paste in the Base 64-encoded public key string, and click **Set**.

![](https://access.redhat.com/webassets/avalon/d/Red_Hat_Enterprise_Linux-7-Linux_Domain_Identity_Authentication_and_Policy_Guide-en-US/images/fbb26af5fd8a911253a61cde7240d3b4/sshkeys-user3.png)

Click **Save** at the top of the page.


-------------------------------
## Hands-On/SSH & UL HPC access

### Step 1a - Connect to UL HPC (Linux / Mac OS / Unix)

Run the following commands in a terminal (substituting *yourlogin* with the login name you received from us):

        (laptop)$> ssh -p 8022 yourlogin@access-iris.uni.lu

Now you probably want to avoid taping this long command to connect to the platform. You can customize SSH aliases for that. Edit the file `~/.ssh/config` (create it if it does not already exist) and adding the following entries:

        Host iris-cluster
            Hostname access-iris.uni.lu
            User yourlogin
            Port 8022
            ForwardAgent no

Now you shall be able to issue the following (simpler) command to connect to the cluster and obtain the welcome banner:

        (laptop)$> ssh iris-cluster



We are now going to illustrate the quick configuration of SSH to facilitate access to the two instances.

```bash
(laptop)$> nano ~/.ssh/config    # OR subl vim emacs  etc...
```

Create the following content:

```bash
# Common options
Host *
    Compression yes
    ConnectTimeout 15

Host iris-cluster
    Hostname access-iris.uni.lu

Host aion-cluster
    Hostname access-aion.uni.lu

# /!\ ADAPT 'yourlogin' accordingly
Host *-cluster
    User yourlogin
    Port 8022
    ForwardAgent no
```

Now you can test the configuration with: `ssh iris-cluster`:

``` bash
(laptop)$> ssh iris-cluster
==================================================================================
 Welcome to access1.iris-cluster.uni.lux
==================================================================================
                            _                         _
                           / \   ___ ___ ___  ___ ___/ |
                          / _ \ / __/ __/ _ \/ __/ __| |
                         / ___ \ (_| (_|  __/\__ \__ \ |
                        /_/   \_\___\___\___||___/___/_|
               _____      _        ____ _           _          __
              / /_ _|_ __(_)___   / ___| |_   _ ___| |_ ___ _ _\ \
             | | | || '__| / __| | |   | | | | / __| __/ _ \ '__| |
             | | | || |  | \__ \ | |___| | |_| \__ \ ||  __/ |  | |
             | ||___|_|  |_|___/  \____|_|\__,_|___/\__\___|_|  | |
              \_\                                              /_/
==================================================================================

=== Computing Nodes ========================================= #RAM/n === #Cores ==
 iris-[001-108] 108 Dell C6320 (2 Xeon E5-2680v4@2.4GHz [14c/120W]) 128GB  3024
 iris-[109-168]  60 Dell C6420 (2 Xeon Gold 6132@2.6GHz [14c/140W]) 128GB  1680
 iris-[169-186]  18 Dell C4140 (2 Xeon Gold 6132@2.6GHz [14c/140W]) 768GB   504
                +72 GPU  (4 Tesla V100 [5120c CUDA + 640c Tensor])   16GB +368640
 iris-[187-190]   4 Dell R840 (4 Xeon Platin.8180M@2.5GHz [28c/205W]) 3TB   448
 iris-[191-196]   6 Dell C4140 (2 Xeon Gold 6132@2.6GHz [14c/140W]) 768GB   168
                +24 GPU  (4 Tesla V100 [5120c CUDA + 640c Tensor])   32GB +122880
==================================================================================
  *** TOTAL: 196 nodes, 5824 cores + 491520 CUDA cores + 61440 Tensor cores ***

 Fast interconnect using InfiniBand EDR 100 Gb/s technology
 Shared Storage (raw capacity): 2180 TB (GPFS) + 1300 TB (Lustre) = 3480 TB

 Support (in this order!)                       Platform notifications
   - User DOC ........ https://hpc.uni.lu/docs    - Twitter: @ULHPC
   - FAQ ............. https://hpc.uni.lu/faq
   - Mailing-list .... hpc-users@uni.lu
   - Bug reports .NEW. https://hpc.uni.lu/support (Service Now)
   - Admins .......... hpc-team@uni.lu (OPEN TICKETS)

    ULHPC user guide 2020 available on hpc.uni.lu:
          https://hpc.uni.lu/blog/2020/ulhpc-user-guide-2020/
==================================================================================
 /!\ NEVER COMPILE OR RUN YOUR PROGRAMS FROM THIS FRONTEND !
     First reserve your nodes (using srun/sbatch(1))
[yourlogin@access1 ~]$
```


In the following sections, we assume these aliases to be defined.


### Step 1b - Optional - using SSH proxycommand setup to access the clusters despite port filtering (Linux / Mac OS / Unix)

It might happen that the port 8022 is filtered from your working place. You can easily bypass this firewall rule using an SSH proxycommand to setup transparently multi-hop connexions *through* one host (a gateway) to get to the access frontend of the cluster, as depited below:

    [laptop] -----||--------> 22 [SSH gateway] ---------> 8022 [access-iris]
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
      ProxyCommand ssh -q -x gw -W `basename %h .ulhpc`:%p

Ensure you can connect to the gateway:

    (laptop)$> ssh gw
    (gateway)$> exit # or CTRL-D

The `.ulhpc` suffix we mentioned in the previous configuration is an arbitrary suffix you will now specify in your command lines in order to access the UL HPC platform via the gateway as follows:

    (laptop)$> ssh iris.ulhpc


### Step 1c - Connect to UL HPC (Windows)

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
  * Click on **OK**


### Step 2 - Hands-on/ Transferring files

Directories such as `$HOME`, `$WORK` or `$SCRATCH` are shared among the nodes of the cluster that you are using (including the front-end) via shared filesystems (NFS, Lustre) meaning that:

* every file/directory pushed or created on the front-end is available on the computing nodes
* every file/directory pushed or created on the computing nodes is available on the front-end


#### Step 2a - Linux / OS X / Unix command line tools

The two most common tools you can use for data transfers over SSH:

* `scp`: for the full transfer of files and directories (only works fine for single files or directories of small/trivial size)
* `rsync`: a software application which synchronizes files and directories from one location to another while minimizing data transfer as only the outdated or inexistent elements are transferred (practically required for lengthy complex transfers, which are more likely to be interrupted in the middle).

Of both, normally the second approach should be preferred, as more generic; note that, both ensure a secure transfer of the data, within an encrypted tunnel.

* Create a new directory on your local machine and download a file to transfer (next-gen sequencing data from the NIH Roadmap Epigenomics Project):

		(laptop)$> mkdir file_transfer
		(laptop)$> cd file_transfer
		(laptop)$> wget "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM409nnn/GSM409307/suppl/GSM409307_UCSD.H1.H3K4me1.LL228.bed.gz"

* Transfer the file with scp:

		(laptop)$> scp GSM409307_UCSD.H1.H3K4me1.LL228.bed.gz iris-cluster:

* Connect to the cluster, check if the file is there and delete it.

		(laptop)$> ssh iris-cluster
		(access-iris)$> ls
		(access-iris)$> rm GSM409307_UCSD.H1.H3K4me1.LL228.bed.gz
		rm: remove regular file `GSM409307_UCSD.H1.H3K4me1.LL228.bed.gz'? y
		(access-iris)$> exit

* Transfer the directory with rsync:

		(laptop)$> cd ..
		(laptop)$> rsync -avzu file_transfer iris-cluster:
        (laptop)$> rsync -e "ssh -F <path/to/config>" -avzu . <name>:path/to/remote/dir # Allow you to use a consistent naming across your server for either SSH or SCP/RSYNC data transfers.


* Delete the file and retrieve it from the cluster:

		(laptop)$> rm file_transfer/GSM409307_UCSD.H1.H3K4me1.LL228.bed.gz
		(laptop)$> rsync -avzu iris-cluster:file_transfer .

* **Bonus**: Check where the file is located on the cluster after the rsync.

You can get more information about these transfer methods in the [file transfer documentation](https://hpc.uni.lu/users/docs/filetransfer.html).


#### Step 2b - Windows MobaXterm file transfer

If you are on Windows, you can directly use MobaXterm to transfer files. Connect to your session (see below on how to configure it). On the right panel you should see an **SFTP** panel opened.

![SFTP on MobaXterm](https://github.com/ULHPC/tutorials/raw/devel/beginners/images/moba_sftp.png)

You have just to drag and drop your files to this panel to transfer files to the cluster. You can try to upload this file [ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM409nnn/GSM409307/suppl/GSM409307_UCSD.H1.H3K4me1.LL228.bed.gz](ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM409nnn/GSM409307/suppl/GSM409307_UCSD.H1.H3K4me1.LL228.bed.gz) (next-gen sequencing data from the NIH Roadmap Epigenomics Project)

To retrieve a file from the cluster, you can right click on it and choose the **Download** option. Please refers to MobaXterm documentation for more informations on the available features.

------------------------
## SSH agent

If you are tired of typing your passphrase, use `ssh-agent` to load your private key

```bash
(laptop)$> ssh-add ~/.ssh/id_rsa
Enter passphrase for ~/.ssh/id_rsa:           # <-- enter your passphrase here
Identity added: ~/.ssh/id_rsa (<login>@<hostname>)

(laptop)$> ssh-add ~/.ssh/id_ed25519
Enter passphrase for ~/.ssh/id_ed25519:       # <-- enter your passphrase here
Identity added: ~/.ssh/id_ed25519 (<login>@<hostname>)
```

On **Ubuntu/WSL**, if you experience issues when using `ssh-add`, you should install the `keychain` package and use it as follows (eventually add it to your `~/.profile`):

```bash
# Installation
(laptop)$> sudo apt install keychain

# Save your passphrase
/usr/bin/keychain --nogui ~/.ssh/id_ed25519    # (eventually) repeat with ~/.ssh/id_rsa
# Load the agent in your shell
source ~/.keychain/$(hostname)-sh
```

You are encouraged to **ALWAYS** simplify the SSH connections by embedding them in a dedicated Host entry, allowing to connect to your remote servers with

        ssh [-F <path/to/config>] <name>

```bash
# Now you can connect transparently to both instances
(laptop)$> ssh -F ~/.ssh/config iris-cluster
Welcome to access1.iris-cluster.uni.lux
[...]
[yourlogin@access1 ~]$ logout    # OR CTRL-D

(laptop)$> ssh -F ~/.ssh/config aion-cluster
Welcome to access1.aion-cluster.uni.lux
[...]
[yourlogin@access1 ~]$ logout    # OR CTRL-D
```

-----------------------------------
## SOCKS 5 Proxy plugin (Optional)

Many Data Analytics framework involves a web interface (at the level of the master and/or the workers) you probably want to access in a relative transparent way.

For that, a convenient way is to rely on a SOCKS proxy, which is basically an SSH tunnel in which specific applications forward their traffic down the tunnel to the server, and then on the server end, the proxy forwards the traffic out to the general Internet.
Unlike a VPN, a SOCKS proxy has to be configured on an app by app basis on the client machine, but can be set up without any specialty client agents.

__Setting Up the Tunnel__

To initiate such a SOCKS proxy using SSH (listening on `localhost:1080` for instance), you simply need to use the `-D 1080` command line option when connecting to a remote server:

```bash
(laptop)$> ssh -D 1080 -C <name>
```

* `-D`: Tells SSH that we want a SOCKS tunnel on the specified port number (you can choose a number between 1025-65536)
* `-C`: Compresses the data before sending it
* `-name`: Server name

__Configuring Firefox to Use the Tunnel__

Now that you have an SSH tunnel, it's time to configure your web browser (in this case, Firefox) to use that tunnel.
In particular, install the [Foxy Proxy](https://getfoxyproxy.org/order/?src=FoxyProxyForFirefox)
extension for Firefox and configure it to use your SOCKS proxy:

* Right click on the fox icon
* Options
* **Add a new proxy** button
* Name: `ULHPC proxy`
* Informations > **Manual configuration**
    -  Host IP: `127.0.0.1`
    -  Port: `1080`
    -  Check the **Proxy SOCKS** Option
* Click on **OK**
* Close
* Open a new tab
* Click on the Fox
* Choose the **ULHPC proxy**

You can now access any web interface deployed on any service reachable from the SSH jump host _i.e._ the ULHPC login node.
