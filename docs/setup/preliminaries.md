
`/!\ IMPORTANT`: this is a set of **hands-on** tutorials: participants are expected to bring a laptop and pre-install software _in advance_ to make the best use of time during the proposed tutorials.
If for some reason you are unable to fulfill this pre-requisite, try to seat close to an attendee that is able to perform these tasks.

_Note_: in the following instructions, terminal commands are prefixed by a virtual prompt `$>`which obviously **does not** belong to the command.

## Online accounts

Kindly create in advance the various accounts for the **cloud services** we might use, _i.e._:

* [Github](https://github.com/):
* [Vagrant Cloud](https://vagrantcloud.com/)
* [Docker Hub](https://hub.docker.com/)

## UL HPC account

You need to have an account on our platform. See <https://hpc-docs.uni.lu/accounts/>

## Software List

The following software should be installed, depending on your running platform. Detailed instructions for each OS are depicted below.

| Platform      | Software                                                                                        | Description                                             | Usage               |
|---------------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------|---------------------|
| Mac OS        | [Homebrew](http://brew.sh/)                                                                     | The missing package manager for macOS                   | `brew install ...`  |
| Mac OS        | [iTerm2](https://www.iterm2.com/)                                                               | enhanced Terminal                                       |                     |
| Windows       | [Chocolatey](https://chocolatey.org/)                                                           | Package Manager for Windows                             | `choco install ...` |
| Windows       | [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10) (WSL) | Emulation-like translation of Linux kernel system calls |                     |
| Windows       | [Ubuntu over WSL](https://www.microsoft.com/en-us/p/ubuntu/9nblggh4msv6)                        | Linux Ubuntu on Windows (recommended)                   |                     |
| Windows       | [Windows Terminal](https://github.com/microsoft/terminal)                                       |                                                         |                     |
| Windows       | [MobaXTERM](https://mobaxterm.mobatek.net/)                                                     | Terminal with tabbed SSH client                         |                     |
| Windows       | [SourceTree](https://www.sourcetreeapp.com/)                                                    | _(optional)_ enhanced git GUI                           |                     |
| Windows/Linux | [Virtual Box](https://www.virtualbox.org/)                                                      | Free hypervisor provider for Vagrant                    |                     |
| Windows/Linux | [Vagrant](https://www.vagrantup.com/downloads.html)                                             | Reproducible environments made easy.                    |                     |
| Linux         | Docker for [Ubuntu](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)        | Lightweight Reproducible Containers                     |                     |
| Windows       | [Docker for Windows](https://docs.docker.com/engine/installation/windows/#/docker-for-windows)  | Lightweight Reproducible Containers                     |                     |

Follow the below **custom** instructions depending on your running platform and Operating System.

### Microsoft Windows

#### Chocolatey: The Package Manager for Windows

Follow Installation instructions on <https://chocolatey.org/> - install it as an **administrator** PowerShell.
You'll probably need to reboot your laptop.

__Chocolatey Installation__

With PowerShell, you must ensure Get-ExecutionPolicy is not Restricted.

- Right click on the Windows starting boutton and choose Windows PowerShell

- Run these three commands

```bash
Get-ExecutionPolicy
### if it returns  Restricted then go to the next step
Set-ExecutionPolicy AllSigned   ## or Set-ExecutionPolicy Bypass -Scope Process
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
```
- Then use __chocolatey__ to install a software

```bash
choco.exe install virtualbox
```

#### WSL and Ubuntu over Windows

The [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about) makes this all possible by providing emulation-like translation of Linux kernel system calls to the Windows kernel.
So when a Linux app like Bash executes and makes system calls, the Windows Subsystem for Linux translates those calls into Windows system calls and Bash executes just like on Linux.

* Resources:
    - [Install the Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10)
    - [Run Bash, SSH and other Linux Apps on Windows 10](https://tweaks.com/windows/67183/run-bash-ssh-and-other-linux-apps-on-windows-10/)
    - [Using WSL and MobaXterm to Create a Linux Dev Environment on Windows](https://nickjanetakis.com/blog/using-wsl-and-mobaxterm-to-create-a-linux-dev-environment-on-windows)

__WSL Installation__

In the Start Menu: select "Turn Windows features on or off"

- Tick "Windows Subsystem for Linux"
- reboot to complete the install

![](https://cdn.tweaks.com/img/article/UbuntuOnWindows1.png)

Then  __Enable the Developer mode__

- In Start Menu: select "Developers Settings"
- Turn on Developer Mode

Now you can **Install Ubuntu within the [Microsoft Store](https://aka.ms/wslstore)**

![](https://docs.microsoft.com/en-us/windows/wsl/media/store.png)

* Search for Ubuntu and install it
    - create a new UNIX username (prefer to follow the guidelines on ULHPC facility: first letter of your firstname, followed by your lastname, all in lower case. _Ex for John Doe_: `jdoe`)
    - add the new password
    - Your windows system drive are exposed in the `/mnt/` directory
         * Ex: `C:\Users\MyLogin\Downloads\File.txt` will be located under `/mnt/c/Users/MyLogin/Downloads/File.txt`

**BEWARE** that your Linux file system is stored **in a hidden folder**, under `%userprofile%\AppData\Local\Packages`.
In particular for Ubuntu, the files are located under the `CanonicalGroupLimited.UbuntuonWindows_<hash>\LocalState\rootfs` folder.
**Note that you're NOT supposed to tamper these files**.
But, if you need to view or back up some files, you'll find them stored in a hidden folder.

__Ubuntu Image Customization__

In the Ubuntu bash, you may want to install [of-my-zsh](https://ohmyz.sh/) and the [Powerlevel10k](https://github.com/romkatv/powerlevel10k#oh-my-zsh) prompt following the following guide:

* [Setting up Windows Subsystem for Linux with zsh + oh-my-zsh](https://blog.joaograssi.com/windows-subsystem-for-linux-with-oh-my-zsh-conemu/)

You will need to enable by default the good font (top left window icon / Properties / Fonts)

#### Microsoft Terminal

You probably want to install then  [Windows Terminal](https://github.com/microsoft/terminal)
It offers the ability to use multiple shell environment in one terminal.

* Install the [Windows Terminal from the Microsoft Store](https://aka.ms/terminal). [Other useful link](https://ohmyposh.dev/docs/windows)
* Install the [Cascadia Code PL](https://github.com/microsoft/cascadia-code/releases) font
* Install the [Meslo LGM NF](https://ohmyposh.dev/docs/fonts) font
* [Other useful link](https://ohmyposh.dev/docs/windows)


__Changing the Powershell prompt__

Install posh-git and oh-my-posh:

```bash
Install-Module posh-git -Scope CurrentUser
Install-Module oh-my-posh -Scope CurrentUser
Install-Module -Name PSReadLine -Scoope CurrentUser -Force -SkipPublisherCheck 
Set-PoshPrompt -Theme Agnoster
```
If you get an error message "Running scripts is disabled on this system", you have to change the PowerShell execution policy which doesn't allow to run scripts:

```bash
Get-ExecutionPolicy -List
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

To make these changes permanent, append the following lines in the Powershell profile (run `notepad $PROFILE`):

```bash
Import-Module posh-git
Import-Module oh-my-posh
Set-PoshPrompt -Theme Agnoster
```

You will also need to enable the [Cascadia Code PL](https://github.com/microsoft/cascadia-code/releases) font by adding into the Windows Terminal Parameters the following lines in the `settings.json` files under the default

```json
"defaults":
{
    "font": 
            {
                "face": "MesloLGM NF"
            }
        },
        "list": 
        [
            {
                "commandline": "powershell.exe",
                "guid": "{61c54bbd-c2c6-5271-96e7-009a87ff44bf}",
                "hidden": false,
                "name": "Windows PowerShell"
                "fontFace": "MesloLGM NF",
            },
        ]
},
```

#### MobaXterm to Run Graphical Linux Apps

Now you can install MobaXterm using [chocolatey]( https://chocolatey.org/) (within an _administrator_ Powershell -- You may need to enable the appropriate powerline font by defaults in the Properties of the Administrator Powershell):

```bash
$ choco.exe install mobaxterm          # Enhanced X11 Terminal for Windows
$ choco.exe install vcxsrv
```

Alternatively, you can consider using [VcXsrv](https://sourceforge.net/projects/vcxsrv/) as an X-server yet our training sessions will assume you rely on MobaXterm.
However if you wish to install [VcXsrv](https://sourceforge.net/projects/vcxsrv/) to natively run X application from the windows terminal, proceed as follows:

* Run `XLaunch` Wizard
* Accept the default options **BUT** pay attention to **Save** the configuration file at the last step in `%appdata%\Microsoft\Windows\Start Menu\Programs\Startup` (eventually search for the 'Start Menu' directory in the explorer).
    - **DENY** (cancel) the Firewall request

Every time you run MobaXterm, make sure that the X server is running - check it by clicking on the associate button.
Select the WSL/Ubuntu profile. We need to configure WSL to send the display of its graphical apps over to Windows. Failing to do so will result in the fact that graphical apps will attempt to load natively inside of WSL and nothing will show up.

To do that, we need to set the `DISPLAY` environment variable within WSL. Run `nano ~/.profile` to append the following lines:

```bash
# Check WSL version with 'wsl.exe -l -v'
# If using WSL 1:
export DISPLAY=:0
# If using WSL 2:
export DISPLAY="$(/sbin/ip route | awk '/default/ { print $3 }'):0"
```

#### IDE / Programming Editor

You probably want to install the following editors/IDE:

* [Sublime Text 3](https://www.sublimetext.com/3) -- see this [configuration of Sublime Text](https://nickjanetakis.com/blog/25-sublime-text-3-packages-for-polyglot-programmers) inside of WSL
    - In MobaXterm, follow the instructions provided below in the [Linux section](#linux-debian-ubuntu)
    - Test it with `subl .`
    - Check out [this Sublime Text 3 package configuration](https://github.com/nickjj/sublime-text-3-packages)

* [VSCode](https://code.visualstudio.com/) (Visual Studio Code)
* [PyCharm](https://www.jetbrains.com/pycharm/)

#### Useful applications install

Then, while most of the below software are covered in the trainings, if you want a fast setup, once you have Chocolatey installed, run the following within an _administrator_ Powershell:

~~~bash
$ choco.exe install git gitflow-avh    # (newer) Git stuff
$ choco.exe install mobaxterm          # Enhanced X11 Terminal for Windows
$ choco.exe install virtualbox         # install virtualbox -- see https://www.virtualbox.org/
$ choco.exe install vagrant            # install Vagrant    -- see https://www.vagrantup.com/downloads.html
$ choco.exe install docker-desktop     # install Docker -- https://docs.docker.com/engine/installation/mac/
~~~

#### Update to WSL 2

Better performances of your Linux subsystem can be obtained by migrating to WSL 2.

In an administrator Powershell, enable the Virtual Machine Platform

```bash
Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
```

Then you can get the list of your WSL systems and convert them as follows:

```bash
$ wsl -l -v
  NAME     STATE     VERSION
* Ubuntu   Stopped   1
# Convert -- Adapt the Distribution name accordinaly
$ wsl --set-version Ubuntu 2
```

For setting all future distributions to use WSL 2, you will need to use the following command:

```bash
$ wsl --set-default-version 2
```

Finally, verify that your changes worked:

```bash
$ wsl -l -v    # OR wsl --list --verbose
  NAME     STATE     VERSION
* Ubuntu   Stopped   2
```


### Mac OS X

* Resources:
    - [Configuring Mac OS](https://varrette.gforge.uni.lu/blog/2017/01/17/configuring-mac-os-on-your-brand-new-laptop/)

Install [iterm2](https://iterm2.com/) and [Homebrew](https://brew.sh/)
Once you have [Homebrew](http://brew.sh/) installed:

~~~bash
$ brew install git git-flow    # (newer) Git stuff
$ brew install mkdocs               # (optional) install mkdocs
$ brew install pyenv pyenv-virtualenv direnv # see https://varrette.gforge.uni.lu/tutorials/pyenv.html
$ brew install virtualbox      # install virtualbox -- see https://www.virtualbox.org/
$ brew install vagrant         # install Vagrant    -- see https://www.vagrantup.com/downloads.html
$ brew install vagrant-manager # see http://vagrantmanager.com/
$ brew install docker          # install Docker -- https://docs.docker.com/engine/installation/mac/
# Note that you probably want to install Firefox, Chrome etc. with brew
$ brew install firefox
$ brew install google-chrome
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

### Linux (Debian / Ubuntu)

~~~bash
# Adapt the package names (and package manager) in case you are using another Linux distribution.
$ sudo apt-get update
$ sudo apt-get install git git-flow build-essential
$ sudo apt-get install rubygems virtualbox vagrant virtualbox-dkms
~~~

For [Docker](https://docker.com/), choose your distribution from https://docs.docker.com/engine/installation/linux/
and follow the instructions.
You need a reasonably new kernel version (3.10 or higher).
Here are detailed instuctions per OS:

* [Ubuntu](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)
* [Debian](https://docs.docker.com/engine/installation/linux/docker-ce/debian/)
* [CentOS](https://docs.docker.com/engine/installation/linux/docker-ce/centos/)

You may want to install Sublime Text:

```bash
wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add -
sudo apt-get install apt-transport-https
echo "deb https://download.sublimetext.com/ apt/stable/" | sudo tee /etc/apt/sources.list.d/sublime-text.list
sudo apt-get update
sudo apt-get install sublime-text
```

Check out [this Sublime Text 3 package configuration](https://github.com/nickjj/sublime-text-3-packages):

```bash
git clone https://github.com/nickjj/sublime-text-3-packages.git ~/.config/sublime-text-3
```

Then test the configuration with `subl .`


## Post-Installations checks

__Git__:

(Eventually) Make yourself known to Git

~~~bash
$ git config –-global user.name  "Firstname LastName"              # Adapt accordingly
$ git config –-global user.email "Firstname.LastName@domain.org"   # Adapt with your mail
# Eventually, if you have a GPG key, use the public key to sign your commits/tags
$ git config --global user.helper osxkeychain       # Only on Mac OS
$ git config --global user.signingkey <fingerprint> # Ex: git config --global user.signingkey 5D08BCDD4F156AD7
# you can get your key fingerprint (prefixed by 0x) with 'gpg -K --fingerprint | grep sec'
~~~


To clone and install this repository, follow the [installation instructions](install.md).


__Vagrant__

Ensure that vagrant is running and has the appropriate plugins from the command line

```bash
$ vagrant --version
Vagrant 2.2.13
```

__Docker (only required for containers tutorials)__

Launch the `Docker` app and then check that the [Docker](https://www.docker.com/) works:

~~~bash
$ docker info
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
$ docker pull centos
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
