[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/linux-shell/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/linux-shell/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/linux-shell/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)                                                                                                                                                                                                           

# Introducing the UNIX/Linux Shell

     Copyright (c) 2013-2021 UL HPC Team <hpc-sysadmins@uni.lu>

The use of the shell is fundamental to a wide range of advanced computing tasks, including high-performance computing. These lessons will introduce you to this powerful tool.
We will cover a range of basic commands that will help you navigate and explore, create files and directories, write script, read and concatenate, how to use command arguments/options and combine existing commands, as well as copy, move and remove files.

 [![](https://github.com/ULHPC/tutorials/raw/devel/linux-shell/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/linux-shell/preliminaries/slides.pdf)

## Very first steps on UNIX/Linux Shell and Command lines

If you're not already comfortable manipulating files and directories, searching for files with `grep` and `find`, and writing simple loops and scripts. Please follow this lesson [Unix Shell from Software Carpentry](https://swcarpentry.github.io/shell-novice/) (from which we made our slides).

You'll learn within a command shell to:

* move around on your computer
* see what files and directories you have
* specify the location of a file or directory on your computer
* create, copy, and delete files and directories
* edit files
* combine existing commands to do new things
* perform the same actions on many different files
* save and re-use commands
* find files and find things in them
* write scripts

Having this basic UNIX/Linux Shell knowledge is the first requirement to use efficiently an HPC facility.

## Editors
### Nano

`$ nano <path/filename>`

* quit and save: `CTRL+x`
* save: `CTRL+o`
* highlight text: `Alt+a`
* Cut the highlighted text: `CTRL+k`
* Paste: `CTRL+u`


### Vim

`$ vim <path/filename>`

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

## Persistent Terminal Sessions using GNU Screen

[GNU Screen](http://www.gnu.org/software/screen/) is a tool to manage persistent terminal sessions.
It becomes interesting since you will probably end at some moment with the following  scenario:

> you frequently program and run computations on the UL HPC platform _i.e_ on a remote Linux/Unix computer, typically working in six different terminal logins to the access server from your office workstation, cranking up long-running computations that are still not finished and are outputting important information (calculation status or results), when you have 2 interactive jobs running... But it's time to catch the bus and/or the train to go back home.

Probably what you do in the above scenario is to

a. clear and shutdown all running terminal sessions

b. once at home when the kids are in bed, you're logging in again... And have to set up the whole environment again (six logins, 2 interactive jobs etc. )

c. repeat the following morning when you come back to the office.

Enter the long-existing and very simple, but totally indispensable [GNU screen](http://www.gnu.org/software/screen/) command. It has the ability to completely detach running processes from one terminal and reattach it intact (later) from a different terminal login.

### Pre-requisite: screen configuration file `~/.screenrc`

While not mandatory, we advise you to rely on our customized configuration file for screen [`.screenrc`](https://github.com/ULHPC/dotfiles/blob/master/screen/.screenrc) available on [Github](https://github.com/ULHPC/dotfiles/blob/master/screen/.screenrc).

Otherwise, simply clone the [ULHPC dotfile repository](https://github.com/ULHPC/dotfiles/) and make a symbolic link `~/.screenrc` targeting the file `screen/screenrc` of the repository.

### Screen commands

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
