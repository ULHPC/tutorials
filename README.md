-*- mode: markdown; mode: auto-fill; fill-column: 80 -*-
`README.md`

Copyright (c) 2013 [Sebastien Varrette](mailto:<Sebastien.Varrette@uni.lu>) [www](http://varrette.gforge.uni.lu)

        Time-stamp: <Dim 2013-11-10 21:28 svarrette>

-------------------

# UL HPC Tutorials 

## Synopsis

This repository holds a set of tutorials to help the users of the
[UL HPC](https://hpc.uni.lu) platform to better understand or simply use our
platform. 

## Tutorials layout

For each subject, you will have a dedicated directory organized as follows:

* `README.md`: hold the description of the tutorial
* `Makefile`: a [GNU make](http://www.gnu.org/software/make/) configuration
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


# Proposing a new tutorial / Contributing to this repository 

## Pre-requisites

### Git

You should become familiar (if not yet) with Git. Consider these resources:

* [Git book](http://book.git-scm.com/index.html)
* [Github:help](http://help.github.com/mac-set-up-git/)
* [Git reference](http://gitref.org/)

### git-flow

The Git branching model for this repository follows the guidelines of [gitflow](http://nvie.com/posts/a-successful-git-branching-model/).
In particular, the central repo (on `github.com`) holds two main branches with an infinite lifetime:

* `production`: the *production-ready* tutorials
* `devel`: the main branch where the latest developments interviene. This is the
  *default* branch you get when you clone the repo. 

### Local repository setup

This repository is hosted on out [GitHub](https://github.com/ULHPC/tutorials).
Once cloned, initiate the potential git submodules etc. by running: 

    $> cd tutorials
    $> make setup

## Fork this repository

You shall now
[fork this repository](https://help.github.com/articles/fork-a-repo). 

Try to be compliant with the above mentioned rules.
