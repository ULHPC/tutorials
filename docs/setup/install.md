First of all, ensure you have installed the [Pre-requisites / Preliminary software](preliminaries.md) and followed the corresponding configuration.

Then this repository is hosted on [Github](https://github.com/ULHPC/tutorials).
Assuming you have installed `git`:

To clone this repository, proceed as follows (adapt accordingly):

```bash
$> mkdir -p ~/git/github.com/ULHPC
$> cd ~/git/github.com/ULHPC
$> git clone https://github.com/ULHPC/tutorials.git
$> cd tutorials
$> make setup          # Initiate git submodules etc...
```

You'll probably wish to have a separate directory structure when working in this tutorial. Here is a suggested approach:

```bash
$> mkdir -p ~/tutorials/HPC-School-2018      # Adapt accordingly
$> cd ~/tutorials/HPC-School-2018
$> ln -s ~/git/github.com/ULHPC/tutorials  ref.d
```

The `make setup` step is **required**.
This will initiate the [Git submodules of this repository](.gitmodules) and setup the [git flow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) layout for this repository.

Note that later on, you can upgrade the [Git submodules](.gitmodules) to the latest version by running:

    $> make upgrade

If upon pulling the repository, you end in a state where another collaborator have upgraded the Git submodules for this repository, you'll end in a dirty state (as reported by modifications within the `.submodules/` directory). In that case, just after the pull, you **have to run** the following to ensure consistency with regards the Git submodules:

    $> make update
