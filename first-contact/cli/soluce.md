# Solutions of cli exercises

## Moving through the file system exercise {: #fs}

* Type `cd`
  * You should see `0 [<your_username>@access1 ~]$`. Note the `~` symbol in the prompt. It shows that you are in your home directory.
* Type `cd ./hpc-school-for-beginners/CLI/playground/`
  * You should see `0 [<your_username>@access1 playground]$`. Type `pwd` and you will see that you are in `/home/users/<your_username>/hpc-school-for-beginners/CLI/playground`
* Type `cd ../docs`. You should see that are in the docs directory.


## Command help exercise {: #help}

* What does the **echo** command do?

Use the man command:

```bash
$ man echo 
```

Output (partial):

```bash
ECHO(1)                                                                                           User Commands                                                                                           ECHO(1)

NAME
       echo - display a line of text

SYNOPSIS
       echo [SHORT-OPTION]... [STRING]...
       echo LONG-OPTION

DESCRIPTION
       Echo the STRING(s) to standard output.

```

We can see that echo simply repeats a string passed as an argument. If I type `$ echo echo`, it will write *echo* in the terminal.  
This command can be useful in scripts as debug information or to display system variables.

* What does the **df-ulhpc** command do?

The df-ulhpc command does not provide a man page. We can however try to use the --help flag.

```bash
$ df-ulhpc --help
```

Output (partial):
```plain
NAME
  df-ulhpc

SYNOPSIS
  df-ulhpc [-h] [-i]

DESCRIPTION

  df-ulhpc outputs a table containing the list of directories accessible by the user
  and the state of the quota.
```

We can see that the command's help looks a lot like a man page. It shows what directories to which you have access and you storage quotas.
