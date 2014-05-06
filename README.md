-*- mode: markdown; mode: auto-fill; fill-column: 80 -*-
`README.md`

Copyright (c) 2013 [Sebastien Varrette](mailto:<Sebastien.Varrette@uni.lu>) [www](http://varrette.gforge.uni.lu)

        Time-stamp: <Mar 2014-05-06 13:11 svarrette>

-------------------

# UL HPC Tutorials 

## Synopsis

This repository holds a set of tutorials to help the users of the
[UL HPC](https://hpc.uni.lu) platform to better understand or simply use our
platform. 

## Tutorials layout

For each subject, you will have a dedicated directory organized as follows:

* `README.md`: hold the description of the tutorial
* (eventually) `Makefile`: a [GNU make](http://www.gnu.org/software/make/) configuration
  file, consistent with the following conventions for the lazy users who do not
  wish to do all the steps proposed in the tutorial: 
  
  * `make fetch` will retrieve whatever archives / sources are required to
    perform the tutorial
  * `make build` will automate the build of the software
  * `make run_interactive` to perform a sample run in interactive mode
  * `make run` to perform a sample run in passive mode (_i.e_  via `oarsub -S
    ...` typically)
  * (eventually) `make plot` to draw a plot illustrating the results obtained
    through the run.

## List of proposed tutorials (cf [UL HPC School](http://hpc.uni.lu/hpc-school/)):

__Basic tutorials__:

* PS1: [Getting started](https://github.com/ULHPC/tutorials/tree/devel/basic/getting_started)
* PS2: [HPC workflow with sequential jobs](https://github.com/ULHPC/tutorials/tree/devel/basic/sequential_jobs)

__Advanced tutorials__:

* PS3: [running the OSU Micro-Banchmarks](https://github.com/ULHPC/tutorials/tree/devel/advanced/OSU_MicroBenchmarks)
* PS3: [running HPL](https://github.com/ULHPC/tutorials/tree/devel/advanced/HPL)
* PS4: [Direct, Reverse and parallel Memory debugging with TotalView](https://github.com/ULHPC/tutorials/tree/devel/advanced/TotalView))
* PS5: [running MATLAB](https://github.com/ULHPC/tutorials/tree/devel/advanced/MATLAB)
* PS6: [running R](https://github.com/ULHPC/tutorials/tree/devel/advanced/R)
* PS7: [running Bio-informatic softwares](https://github.com/ULHPC/tutorials/tree/devel/advanced/Bioinformatics)

## Proposing a new tutorial / Contributing to this repository 

You're using a specific software on the UL HPC platform not listed in the above
list? Then most probably you

1. developed a set of script to effectively run that software 
2. used to face issues such that you're aware (eventually unconsciously) of
tricks and tips for that specific usage.  

Then your inputs are valuable for the other users and we would appreciate your
help to complete this repository with new topics/entries.

### Pre-requisites

#### Git

You should become familiar (if not yet) with Git. Consider these resources:

* [Git book](http://book.git-scm.com/index.html)
* [Github:help](http://help.github.com/mac-set-up-git/)
* [Git reference](http://gitref.org/)

#### git-flow

The Git branching model for this repository follows the guidelines of [gitflow](http://nvie.com/posts/a-successful-git-branching-model/).
In particular, the central repo (on `github.com`) holds two main branches with an infinite lifetime:

* `production`: the *production-ready* tutorials
* `devel`: the main branch where the latest developments interviene. This is the
  *default* branch you get when you clone the repo. 

#### Local repository setup

This repository is hosted on out [GitHub](https://github.com/ULHPC/tutorials).
Once cloned, initiate the potential git submodules etc. by running: 

    $> cd tutorials
    $> make setup

### Fork this repository

You shall now
[fork this repository](https://help.github.com/articles/fork-a-repo). 

Try to be compliant with the above mentioned rules _i.e._ try to define _at
least_ a `README.md` file (using the
[markdown](http://github.github.com/github-flavored-markdown/) format) in the
appropriate directory containing  your guidelines. 
Remember that they shall be understandable for users having no or very few
knowledge on your topic!

One _proposal_ to organize your tutorial: 

* Select a typical sample example that will be used throughout all the tutorial,
  that is easy to fetch from the official page of the software
* dedicate a section to the running of this example in an _interactive_ job such
  that the reader has a better understanding of 
  * the involved modules to load 
  * the classical way to execute the software
  * etc. 
* dedicate a second section to the running of the example in a _passive_ job,
  typically providing a generic launcher script adapted to your software. You
  might adapt / extend the
  [UL HPC launcher scripts](https://github.com/ULHPC/launcher-scripts). 
* a last section would typically involves hints / elements to benchmark the
  execution, add tips/tricks to improve the performances (and see
  the effects of those improvements) and have a way to plot the results. 
