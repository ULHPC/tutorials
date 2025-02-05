

# Command Line Interface Crash Course 

[Completion time ~2h]

The goal of this tutorial is to get you acquainted with the command line, teach you how to interact with the cluster and give you the keys to decrypt the commands you will meet wihle using the ULHPC clusters.

This tutorial is quite long, don't forget about the table of content, which can be expanded, on the left side of the screen. 

## Prerequisites

* A terminal emulator or MobaXterm
* The ability to connect to the cluster (an active HPC account and a registered SSH key pair)
* Pull the git repository of the tutorial.
	* connect to the cluster
	* type the following commands:
	```bash
	$ cd
	$ git clone https://github.com/ULHPC/hpc-school-for-beginners.git
	$ cd hpc-school-for-beginners/CLI
	```
	* the commands respectively move you at the root of your home directory, pull the repository, move you the the directory of the tutorial.

## The command line interface in a few words

The Command Line Interface (CLI) is a resource efficient way to interact with a computer. As it requires very little resources, it is well suited to the HPC and is thus the primary mean to interact with the clusters.

When you connect to the cluster (or open aterminal on your own machine), the server will start a **shell**. A shell is a program that allows you to send commands to the server. There are multiple shells which behave a bit differently. The default one on the ULHPC clusters is called **bash**.

As said previously, the shell allows you to send **commands**. These commands are programs that are executed by the server, from moving through the file system to running your experiments.

### The command Prompt

When you connect to the ULHPC clusters, you are greeted with a banner and the message of the day. At the bottom of the screen, you will find a line that looks like the following:

```plain
0 [yourUserName@access1 ~]$
```

Lets decrypt this first line:

* First, there is a number. This correspond to the *exit status* of the previous command. 0 means that the previous command was successful. Anything else indicates that it encountered some error. The error codes are specific to the program that was run. It usually gives some indications on what went wrong but you will have to refer to the program documentation to find its meaning.

* Then your have something between brackets. This can be split in three parts:
	* your username (before the @ symbol)
	* the name of the machine to which you are currently connected (after the @ symbol). In this case, you are on the [access node](../concepts#access-nodes)
	* the current directory, here **~** which is a shorthand for your personnal space or [home directory](../concepts#storage)

* a **$** symbol followed by a blinking caret. This symbol means that the machine is ready to receive your next command. If this symbol is not present, it means a program is currently running and you have to wait for its completion before sending another one. 
In the tutorials, it is common to find the $ symbol before commands to differentiate them from outputs.

During your interactions with the clusters, you will notice that the command prompt sometimes look different:

```plain
0 [yourUserName@aion-0035 docs](87569 1N/T/1CN)$
```
Do not be panic, it does not change anything functionally and only gives you some additional information:
* you can notice that the machine you are connected to is *aion-0035*, which is a [compute node](../concepts#getting-computational-resources)
* you are currently located in the *docs* directory
* some information about your [reservation](../concepts#getting-computational-resources) between parenthesis. Its meaning is out of the scope of this tutorial.

### Anatomy of a command

As we said before, we need to type commands for the server to do something. Luckily, almost all commands follow the same model.

```bash
program [flags]... [arguments]...
```

* *program* is the name of the program you want to start. e.g. cd, matlab, python, ...
* something between *brackets* mean that this part of the command is optional
* an *ellipsis* (...) means that there can be more than one
* Flags and arguments are specific to programs

We will use this notation in the remainder of the tutorial. The linux standard manual pages also follow this notation.

#### Flags (or OPTIONS)

They are options that change the behavior of the program.

* Flags are not sensitive to order
* There are two ways to use flags. Most of them can be written both ways (refer to the program documentation)
	* Long flags - better readability but longer to type
		* start with a double dash, e.g. `--all`
		* can have parameters
			* separated by the equal sign, e.g. `--format=long`
			* or separated by a space, e.g. `--ignore foo`
	* Short flags - easier to type but more confusing
		* start with a single dash symbol, e.g. `-l -a`
		* can be combined, e.g. `-la`. Here again, the order is not relevant `-la` = `-al`
		* can have parameters separated by a space, e.g. `-I foo`

The following commands are equivalent:  
```bash
$ ls -la
$ ls -a -l
$ ls --format=long -a
```

#### Arguments

Arguments are the parameters of the program

* They are positional, i.e. switching them will change the result given by the program. e.g. `cp source destination`
* The number of arguments is specific to each program

The following commands are not equivalent:  
```bash
$ myCommand input.txt output.txt
$ myCommand output.txt input.txt
```

### General tips and tricks

#### Help, I am stuck!
You typed a command and you are stuck without being able to use the terminal.

* The command might take a long time or might be stuck
	* Try the *ctrl+c* combination which interrupts the currently running process
* The command launched a text based interface
	* Usually, you can find on the screen a hint that gives you a key or a key combination you can input to leave. Usually *q* or *esc*.

#### Autocompletion

The *tab* key tries to autocomplete your current command.

* Try typing the first few letters of a command then hit tab. The shell will either complete the command or provide a list of possibilities.
* When writing the path to a file, tabbing provides a list of valid propositions.

#### History

* Using the up and down arrow keys allow you to go through recent commands you typed.
* The **history** command presents you with the last 1000 commands you typed. Very useful if you forgot how you did something in the past.
* **ctrl+r** allows you to search your history. Useful for long commands you type often.

### A little bit of help

Your next question should be: How can I know which flags and arguments the program I want to run supports?

#### Man: manual pager

Luckily, most linux programs provide a **man** page.  
Here is an abridged version of man page of the **man** command.

```plain
MAN(1)                                                                                          Manual pager utils                                                                                         MAN(1)

NAME
       man - an interface to the system reference manuals

SYNOPSIS
       man [man options] [[section] page ...] ...
       man -k [apropos options] regexp ...
       man -K [man options] [section] term ...

DESCRIPTION
       man is the system's manual pager.  Each page argument given to man is normally the name of a program, utility or function.  The manual page associated with each of these arguments is then found and dis‐
       played.   A  section,  if  provided,  will direct man to look only in that section of the manual.  The default action is to search in all of the available sections following a pre-defined order (see DE‐
       FAULTS), and to show only the first page found, even if page exists in several sections.

EXAMPLES
       man ls
           Display the manual page for the item (program) ls.

OVERVIEW
       Many  options are available to man in order to give as much flexibility as possible to the user.  Changes can be made to the search path, section order, output processor, and other behaviours and opera‐
       tions detailed below.

OPTIONS
       Non-argument options that are duplicated either on the command line, in $MANOPT, or both, are not harmful.  For options that require an argument, each duplication will  override  the  previous  argument
       value.

   General options
       -C file, --config-file=file
              Use this user configuration file rather than the default of ~/.manpath.

       -d, --debug
              Print debugging information.
```

We can see here multiple useful sections:

* NAME - a brief description of what to command does.
* SYNOPSIS - How to use the command. Here we can see that depending on the flag, the command expects different parameters and arguments.
* EXAMPLES - some example showing how the command is used
* OVERVIEW - a more in depth description of the command's behavior
* OPTIONS - a description of the flags/options that can be used with the command

In its most basic usage (and most common), you can use the command as follows:
`$ man myCommand`

**You can quit the man page by hitting 'q'.**

#### Help flag

Some commands do not provide man pages. In this case, your best option is to try the --help flag: 
`$ myCommand --help`

The help flag doesn't necessarily follows the same format as the man page but provides the same kind of information, albeit in a much briefer form.

#### Exercise

* What does the **echo** command do?
* What does the **df-ulhpc** command do?

You can have a look at the solution [here](./soluce.md#help)

## Navigating through the file system

### The linux file system

The Linux file system is a tree like structure. 

Unlike windows, physical drives are not represented in the file system but are abstracted as mount points (i.e. directories). It means that multiple types of storage can be available under separate directories. As an example, our home directory storage is located under `/home/users` our scratch directory under `/scratch`. Those are two physically separate storage systems.

The file system is standardized and heavily based on conventions. As an example: 

* `/etc` contains configuration files
* `/bin` contains binaries (executable programs)
* `/home` contains the users home directories (personal spaces)

You will also encounter the following notations:

* `~` represents your home directory, your personal space. It is a shorthand for `/home/users/<your_user_name>`
* `.` represents the curent directory
* `..` represents the parent directory
* `.foo` represents a hidden file or directory called 'foo'

### Prepare your environment

First, we should place ourselves in the right directory to ensure the following exercices make sense. If your followed the [prerequisites](#Prerequisites), connect to the cluster (iris or aion, it does not matter) with your favorite terminal emulator and  type (or copy/paste) the following command:

```bash
cd ~/hpc-school-for-beginners/CLI/
```

After pressing *return*, you should see the following prompt:

```bash
0 [<your_login>@access1 CLI]$
```

Your are ready to begin.

### File system navigation related commands

#### pwd

`pwd` stands for **p**rint **w**orking **d**irectory, that is the directory in which your are currently located. It is very useful to locate yourself in the file system as the command prompt only displays the current directory and not the complete path.

To use the command, simply type `pwd`. The system should output `/home/users/<your_login/hpc-school-for-beginners/CLI`.

#### ls

`ls` stands for **l**ist **d**irectory. It displays the content of a directory and accepts mutiple options and arguments.

* If I type `ls`, it displays the content of the current directory.
* Using the `-a` flag also shows hidden files and directories (the ones that start with a `.`).
* Using the `-l` flag formats the output differently and shows more details abouth the files (permissions, ownership, modification date, ...).
* Adding a *path* as an argument, lists the files and directories at this location.

As an example, the following command `ls -la` will display

``` bash
drwx------. 5 <your_username> clusterusers 16384 Jan  6 11:43 .
drwxr-xr-x. 6 <your_username> clusterusers   512 Jan  6 11:38 ..
drwxr-xr-x. 2 <your_username> clusterusers   512 Jan  6 08:54 docs
drwxr-xr-x. 2 <your_username> clusterusers   512 Jan  6 08:54 final_boss
drwxr-xr-x. 7 <your_username> clusterusers   512 Jan  6 10:43 playground
```

We can see three directories; docs, final_boss and playground. We can also see the `.` and `..` directories which represent respectively the current and the parent directories.

#### cd

`cd` stands for **c**hange **d**irectory. This commands allows you to move through the file system.

* `cd` with no further argument will send you to your home directory (`0 [<your_username>@access1 ~]$`).
* `cd /some/path`. *cd* with an absolute path (a path that starts with a */*) will move you to this directory if it exists.
* `cd some/path` or `cd ./some/path`. *cd* with a relative path (a path the starts in the current directory) will move you to this directory if it exists. 
* `cd ..` will move you to the parent directory of your current location. `cd ../..` will move you two levels ups.

Notes:
* Relative paths can start with a `.` or directly by the name of a child directory. For the sake of clarity, prefer using the explicit `./` notation.
* When typing a path, you can hit tab 
	* once to autocomplete it if there is only one possible option
	* twice to show the different possibilities if there are multiple options

### Activity

* Go to your home directory
* From there go to hpc-school-for-beginners/CLI/playground
* Go up a level and to the docs directory (in one command)

The solution is [here](./soluce.md#fs).

## Executing programs and scripts

## File manipulation

## Reading and writing files

## File access permissions

## Synchronizing data

### Rsync (Linux and Mac)

### Using mobaxterm

## Test your skills