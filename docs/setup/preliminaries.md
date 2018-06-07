
`/!\ IMPORTANT`: this is a set of **hands-on** tutorials: participants are expected to bring a laptop and pre-install software _in advance_ to make the best use of time during the proposed tutorials.
If for some reason you are unable to fulfill this pre-requisite, try to seat close to an attendee that is able to perform these tasks.

_Note_: in the following instructions, terminal commands are prefixed by a virtual prompt `$>`which obviously **does not** belong to the command.

### Online accounts

Kindly create in advance the various accounts for the **cloud services** we might use, _i.e._:

* [Github](https://github.com/):
* [Vagrant Cloud](https://vagrantcloud.com/)
* [Docker Hub](https://hub.docker.com/)

### UL HPC account

You need to have an account on our platform. See <https://hpc.uni.lu/users/get_an_account.html>

### Software

Install the following software, depending on your running platform:

| Platform      | Software                                                                                       | Description                           | Usage                   |
|---------------|------------------------------------------------------------------------------------------------|---------------------------------------|-------------------------|
| Mac OS        | [Homebrew](http://brew.sh/)                                                                    | The missing package manager for macOS | `brew install ...`      |
| Mac OS        | [Brew Cask Plugin](https://caskroom.github.io)                                                 | Mac OS Apps install made easy         | `brew cask install ...` |
| Mac OS        | [iTerm2](https://www.iterm2.com/)                                                              | _(optional)_ enhanced Terminal        |                         |
| Windows       | [MobaXTERM](https://mobaxterm.mobatek.net/)                                                    | Terminal with tabbed SSH client       |                         |
| Windows       | [Git for Windows](https://git-for-windows.github.io/)                                          | I'm sure you guessed                  |                         |
| Windows       | [SourceTree](https://www.sourcetreeapp.com/)                                                   | _(optional)_ enhanced git GUI         |                         |
| Windows/Linux | [Virtual Box](https://www.virtualbox.org/)                                                     | Free hypervisor provider for Vagrant  |                         |
| Windows/Linux | [Vagrant](https://www.vagrantup.com/downloads.html)                                            | Reproducible environments made easy.  |                         |
| Linux         | Docker for [Ubuntu](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)       | Lightweight Reproducible Containers   |                         |
| Windows       | [Docker for Windows](https://docs.docker.com/engine/installation/windows/#/docker-for-windows) | Lightweight Reproducible Containers   |                         |

Follow the below **custom** instructions depending on your running platform and Operating System.

#### Mac OS X

Once you have [Homebrew](http://brew.sh/) installed:

~~~bash
$> brew install git-core git-flow    # (newer) Git stuff
$> brew install mkdocs               # (optional) install mkdocs
$> brew install pyenv pyenv-virtualenv direnv # see https://varrette.gforge.uni.lu/tutorials/pyenv.html
$> brew tap caskroom/cask            # install brew cask  -- see https://caskroom.github.io/
$> brew cask install virtualbox      # install virtualbox -- see https://www.virtualbox.org/
$> brew cask install vagrant         # install Vagrant    -- see https://www.vagrantup.com/downloads.html
$> brew cask install vagrant-manager # see http://vagrantmanager.com/
$> brew cask install docker          # install Docker -- https://docs.docker.com/engine/installation/mac/
~~~

_Note_: later on, you might wish to use the following shell function to update the software installed using [Homebrew](http://brew.sh/).

```bash
bup () {
	echo "Updating your [Homebrew] system"
	brew update
	brew upgrade
	brew cu
	brew cleanup
	brew cask cleanup
}
```

#### Linux (Debian / Ubuntu)

~~~bash
# Adapt the package names (and package manager) in case you are using another Linux distribution.
$> sudo apt-get update
$> sudo apt-get install git git-flow build-essential
$> sudo apt-get install rubygems virtualbox vagrant virtualbox-dkms
~~~

For [Docker](https://docker.com/), choose your distribution from https://docs.docker.com/engine/installation/linux/
and follow the instructions.
You need a reasonably new kernel version (3.10 or higher).
Here are detailed instuctions per OS:

* [Ubuntu](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)
* [Debian](https://docs.docker.com/engine/installation/linux/docker-ce/debian/)
* [CentOS](https://docs.docker.com/engine/installation/linux/docker-ce/centos/)


#### Windows

On Windows (10, 7/8 should also be OK) you should download and install the following tools:

* [MobaXterm](http://mobaxterm.mobatek.net). You can also check out the [MobaXterm demo](http://mobaxterm.mobatek.net/demo.html) which shows an overview of its features.
    - See also [official ULHPC SSH access instructions](https://hpc.uni.lu/users/docs/access/access_windows.html)

* [VirtualBox](https://www.virtualbox.org/wiki/Downloads), download the latest [VirtualBox 'Windows hosts' installer](https://www.virtualbox.org/wiki/Downloads)  and [VirtualBox Extension Pack](https://www.virtualbox.org/wiki/Downloads) .
    - First, install VirtualBox with the default settings. Note that a warning will be issued that your network connections will be temporarily impacted, you should continue.
    - Then, run the downloaded extension pack (.vbox-extpack file), it will open within the VirtualBox Manager and you should let it install normally.

* [Vagrant](https://www.vagrantup.com/downloads.html), download the latest [Windows (64 bit) Vagrant installer](https://www.vagrantup.com/downloads.html)
    - Proceed with the installation, no changes are required to the default setup.

* [Git](https://git-scm.com/downloads), download the latest [Git installer](https://git-scm.com/download/win)

The Git installation requires a few changes to the defaults, make sure the following are selected in the installer:

   - Select Components: _Use a TrueType font in all console windows)_
   - Adjusting your PATH environment: _Use Git and optional Unix tools from the Windows Command Prompt_
   - Configuring the line ending conversions: _Checkout Windows-style, commit Unix-style line endings)_
   - Configuring the terminal emulator to use with Git Bash: _Use MinTTY (the default terminal of MSYS2)_
   - Configuring extra options: _Enable symbolic links_

Please note that to clone a Git repository which contains symbolic links (symlinks), you **must start a shell** (Microsoft PowerShell in this example, but a Command Prompt - cmd.exe - or Git Bash shell should work out fine) **with elevated (Administrator) privileges**. This is required in order for git to be able to create symlinks on Windows:

* Start Powershell:
    1. In the Windows Start menu, type PowerShell
    2. Normally PowerShell will appear as the first option on the top as **Best match**
    3. Right click on it and select "Run as administrator"

See also the instructions and screenshots provided on this [tutorial](http://rr-tutorials.readthedocs.io/en/latest/setup/#windows).

## Post-Installations checks

__Git__:

(Eventually) Make yourself known to Git

~~~bash
$> git config –-global user.name  "Firstname LastName"              # Adapt accordingly
$> git config –-global user.email "Firstname.LastName@domain.org"   # Adapt with your mail
~~~

Clone the [tutorial repository on Github](https://github.com/ULHPC/tutorials) from a Terminal (Powershell as `administrator` under windows):

~~~bash
$> mkdir -p ~/git/github.com/ULHPC
# Clone reference git
$> cd ~/git/github.com/ULHPC
$> git clone https://github.com/ULHPC/tutorials.git
~~~

__Vagrant__

Ensure that vagrant is running and has the appropriate plugins from the command line

```bash
$> vagrant --version
Vagrant 2.1.1
```

__Docker (only required for containers tutorials)__

Launch the `Docker` app and then check that the [Docker](https://www.docker.com/) works:

~~~bash
$> docker info
Containers: 9
 Running: 0
 Paused: 0
 Stopped: 9
Images: 12
Server Version: 18.03.1-ce
[...]
~~~

*  Pull the docker containers we might need for the concerned tutorial

~~~bash
$> docker pull centos
~~~

* Login onto you [Docker hub account](https://hub.docker.com/) (take note of your Docker Hub ID and password).
    - With docker installed, run

~~~bash
$ docker login -u <your docker hub ID>
~~~
and enter your password.

Note that if the Docker installation fails, you can use <http://play-with-docker.com/> to try Docker, but **it won't work if all of us try it once!**
So use it only as a last resort, and it is up to you to use any important information (like the Docker hub account) inside it.

__Mkdocs__

It probably makes sense to install  [`mkdocs`](http://www.mkdocs.org/#installation)  to be able to generate locally the current documentation.

Follow for that the instructions provided on the [`../rtfd.md`](../rtfd.md).
