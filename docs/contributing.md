# Proposing a new tutorial / Contributing to this repository

You're using a specific software on the UL HPC platform not listed in the above list? Then most probably you

1. developed a set of script to effectively run that software
2. used to face issues such that you're aware (eventually unconsciously) of tricks and tips for that specific usage.

Then your inputs are valuable for the other users and we would appreciate your help to complete this repository with new topics/entries.

To do that, the general approach is similar to the one proposed by [Github via the Forking procedure](https://help.github.com/articles/fork-a-repo/).
Since we use [git-flow](https://github.com/nvie/gitflow), your workflow for contributing to this repository should typically involve the following steps:

1. [Fork it](https://help.github.com/articles/fork-a-repo/)
2. Initialize your local copy of the repository (including git submodules etc.): `make setup`
2. Create your feature branch: `git flow feature start <feature_name>`
3. Commit your changes: `git commit -am 'Added some feature'`
4. Publish your feature branch: `git flow feature publish <feature_name>`
5. Create new [Pull Request](https://help.github.com/articles/using-pull-requests/)

More details are provided below.

## git-flow

The Git branching model for this repository follows the guidelines of [gitflow](http://nvie.com/posts/a-successful-git-branching-model/).
In particular, the central repo (on `github.com`) holds two main branches with an infinite lifetime:

* `production`: the *production-ready* tutorials
* `devel`: the main branch where the latest developments interviene. This is the *default* branch you get when you clone the repo.

## New tutorial layout

So assuming you have [forked this repository](https://help.github.com/articles/fork-a-repo) to work freely on your own copy of it, you can now feed a new tutorial, assuming you follow the below guidelines.

### Directory Layout

```
<topic>/<name>  # Select the appropriate root directory
├── README.md              # Main tutorial file, in Markdown
├── index.md -> README.md  # Symlink (for mkdocs)
├── slides.pdf             # Slides proposing an overview of the tutorial
├── cover_slides.png       # Picture of the cover of the slide
├── Makefile               # GNU Makefile offering the targets 'fetch', 'compile', 'run' and 'plot'
├── plots                  # Directory hosting the Gnuplots / R plots data
├── runs/                  # Directory hosting the data/logs of the runs
├── scripts/               # Eventually, a directory hosting some specific scripts
└── launcher-<name>.{slurm|oar}.sh # launcher script to be used in the tutorial

# Prepare the appropriate link for ReadtheDocs -- if needed
docs/<topic> -> ../<topic>
# such that 'docs/<topic>/<name>' points to '../<topic>/<name>'
```

You SHOULD  stick to a single `README.md` file, (using the [markdown](http://github.github.com/github-flavored-markdown/) format) if possible.

Kindly follow the following format for this file (adapt `path/to` accordingly):

```
[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/path/to/slides.pdf) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/path/to/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/path/to/) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

# <the title>

      Copyright (c) 2013-2018 [You name,] UL HPC Team  <hpc-sysadmins@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/path/to/cover_slides.png)](https://github.com/ULHPC/tutorials/raw/devel/path/to/slides.pdf)

The objective of this tutorial is to cover XXX, in particular:

* objective 1
* objective 2

--------------------
## Pre-requisites ##

Ensure you are able to [connect to the UL HPC clusters](https://hpc-docs.uni.lu/connect/access/)
In particular, recall that the `module` command **is not** available on the access frontends.

Now you'll need to pull the latest changes in your working copy of the [ULHPC/tutorials](https://github.com/ULHPC/tutorials) you should have cloned in `~/git/github.com/ULHPC/tutorials` (see ["preliminaries" tutorial](../../preliminaries/))

``` bash
(access)$> cd ~/git/github.com/ULHPC/tutorials
(access)$> git pull
```

Now **configure a dedicated directory `~/tutorials/<topic>` for this session**

``` bash
# return to your home
(access)$> mkdir -p ~/tutorials/<topic>
(access)$> cd ~/tutorials/<topic>
# create a symbolic link to the reference material
(access)$> ln -s ~/git/github.com/ULHPC/tutorials/<path> ref.d
```

**Advanced users** (_eventually_ yet __strongly__ recommended), create a [Tmux](https://github.com/tmux/tmux/wiki) session (see [Tmux cheat sheet](https://tmuxcheatsheet.com/) and [tutorial](https://www.howtogeek.com/671422/how-to-use-tmux-on-linux-and-why-its-better-than-screen/)) or [GNU Screen](http://www.gnu.org/software/screen/) session you can recover later. See also ["Getting Started" tutorial ](../../beginners/).

``` bash
# /!\ Advanced (but recommended) best-practice:
#     Always work within a TMux or GNU Screen session named '<topic>' (Adapt accordingly)
(access-aion)$> tmux new -s HPC-school   # Tmux
(access-iris)$> screen -S HPC-school     # GNU Screen
#  TMux     | GNU Screen | Action
# ----------|------------|----------------------------------------------
#  CTRL+b c | CTRL+a c   | (create) creates a new Screen window. The default Screen number is zero.
#  CTRL+b n | CTRL+a n   | (next) switches to the next window.
#  CTRL+b p | CTRL+a p   | (prev) switches to the previous window.
#  CTRL+b , | CTRL+a A   | (title) rename the current window
#  CTRL+b d | CTRL+a d   | (detach) detaches from a Screen -
# Once detached:
#   tmux ls  | screen -ls : list available screen
#   tmux att | screen -x  : reattach to a past screen
```

Let's get an interactive jobs:

```bash
### Access to ULHPC cluster (if not yet done)
(laptop)$> ssh aion-cluster    # or iris-cluster
### Have an interactive job
# ... either directly
(access)$> si [...]
# ... or using the HPC School reservation 'hpcschool'if needed  - use 'sinfo -T' to check if active and its name
# (access)$> si --reservation=hpcschool [...]
(node)$>
```

-----------------
## Objective 1 ##

instructions

-----------------
## Objective 2 ##

instructions

## Useful references
```


Remember that they shall be understandable for users having no or very few knowledge on your topic!

One _proposal_ to organize the workflow of your tutorial:

* Select a typical sample example that will be used throughout all the tutorial, that is easy to fetch from the official page of the software. Adapt the `make fetch` directive in your root `Makefile` to perform the corresponding actions.
* (eventually) detail how to build the sources (using [EasyBuild](advanced/Easybuild). Adapt the `make build` accordingly.
* dedicate a section to the running of this example in an _interactive_ job such that the reader has a better understanding of:
   - the involved modules to load
   - the classical way to execute the software
   - etc.
   Adapt also the `make run_interactive` accordingly
* dedicate a second section to the running of the example in a _passive_ job, typically providing a generic launcher script adapted to your software. You might adapt / extend the [UL HPC launcher scripts](https://github.com/ULHPC/launcher-scripts) the same way to extend these tutorials. Adapt also the `make run` accordingly.
* a last section would typically involves hints / elements to benchmark the execution, add tips/tricks to improve the performances (and see the effects of those improvements) and have a way to plot the results.  Adapt the `make plot` accordingly

### Semantic Versionning

The operation consisting of releasing a new version of this repository is automated by a set of tasks within the `Makefile` at the root of this repository.

In this context, a version number have the following format:

      <major>.<minor>.<patch>

where:

* `< major >` corresponds to the major version number
* `< minor >` corresponds to the minor version number
* `< patch >` corresponds to the patching version number

Example: `1.2.0`

The current version number is stored in the file `VERSION`. **DO NOT EDIT THIS FILE**, use the below primitives to affect the number it contains.
For more information on the version, run:

     $> make versioninfo

If a new  version number such be bumped, you simply have to run:

     $> make start_bump_{major,minor,patch}

This will start the release process for you using `git-flow`.
Then, to make the release effective, just run:

     $> make release

This will finalize the release using `git-flow`, create the appropriate tag and merge all things the way they should be.
