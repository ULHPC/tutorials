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
{advanced | basic}/<name>  # Select the appropriate root directory
├── .root -> ../../        # Symlink to the root directory
├── README.md              # Main tutorial file, in Markdown
├── tutorial_<name>.pdf    # Slides proposing an overview of the tutorial
├── Makefile               # GNU Makefile offering the targets 'fetch', 'compile', 'run' and 'run_interactive' 
├── plots                  # Directory hosting the Gnuplots / R plots data
│   ├── Makefile -> .root/.submodules/Makefiles/gnuplot/Makefile # in case of Gnuplot...
├── runs/                  # Directory hosting the logs of the runs
├── scripts/               # Eventually, a directory hosting some specific scripts
└── src/                   
    └── slides/            # bearmer sources of the slides
        ├── Makefile -> .root/.submodules/Makefiles/latex/Makefile
        ├── tutorial_<name>.tex        # Main LaTeX sources
        ├── VERSION -> .root/VERSION   # Version file
        ├── __config.sty               # Specifid configs for the slides
        ├── _style.sty -> .root/.templates/slides/_style.sty
        ├── beamerthemeFalkor.sty -> .root/.submodules/beamerthemeFalkor/beamerthemeFalkor.sty
        ├── figures/                   # Local figures
        └── images -> .root/.templates/slides/images
# Prepare the appropriate link for ReadtheDocs
docs/{advanced | basic}/<name>.md -> ../../../{advanced | basic}/<name>/README.md
```

You shall stick to a single `README.md` file, (using the [markdown](http://github.github.com/github-flavored-markdown/) format).
Remember that they shall be understandable for users having no or very few
knowledge on your topic!

One _proposal_ to organize the workflow of your tutorial: 

* Select a typical sample example that will be used throughout all the tutorial, that is easy to fetch from the official page of the software. Adapt the `make fetch` directive in your root `Makefile` to perform the corresponding actions.
* (eventually) detail how to build the sources (using [RESIF](http://resif.readthedocs.org/en/latest/)). Adapt the `make build` accordingly.
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
