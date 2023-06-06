Reproducible pipelines in R
================

[![By
ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu)
[![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html)
[![GitHub
issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/)
[![](https://img.shields.io/badge/slides-PDF-red.svg)](https://github.com/ULHPC/tutorials/raw/devel/maths/R/targets.pdf)
[![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/tree/devel/maths/R/)
[![Documentation
Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io/en/latest/maths/R/)
[![GitHub
forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

------------------------------------------------------------------------

# R Tutorial

      Copyright (c) 2013-2023 Aurelien Ginolhac, UL HPC Team  <hpc-sysadmins@uni.lu>

[![](https://github.com/ULHPC/tutorials/raw/devel/maths/R/img/cover_slides.png)](https://ulhpc-tutorials.readthedocs.io/en/latest/maths/R/targets.html)


Through this tutorial you will learn how to use the R package [**{targets}**](https://books.ropensci.org/targets/): A function-oriented  [Make](https://www.gnu.org/software/make/)-like workflow manager. 

**Warning**: this tutorial does not focus on the learning of R language. If you’re also looking for a
good tutorial on R’s data structures you can take a look at: [Hadley
Wickham’s page](https://adv-r.hadley.nz/vectors-chap.html). Another
[bookdown](https://bookdown.org/)’s book is available for free: [R for
Data Science](https://r4ds.hadley.nz/) by Hadley Wickham, Mine Çetinkaya-Rundel & Garrett Grolemund.


------------------------------------------------------------------------

### Pre-requisites

Ensure you are able to [connect to the UL HPC
clusters](https://hpc-docs.uni.lu/connect/ssh/)

**you MUST work on a computing node**

``` bash
# /!\ FOR ALL YOUR COMPILING BUSINESS, ENSURE YOU WORK ON A COMPUTING NODE
(access-iris)$> si -c 2 -t 1:00:00
```

### On HPC, using Singularity

A [Singularity](https://sylabs.io/docs/) image was prepared that contains an Ubuntu-based R 4.2.2
but with all necessary packages. Of note this was created using `renv::restore()`. [`renv`](https://rstudio.github.io/renv/articles/renv.html) helps to manage R dependency to projects.

- Once on a node, load Singularity

``` bash
module load tools/Singularity

```

### Cloning the demo repository

The demo repository is [targets_demos](https://gitlab.lcsb.uni.lu/aurelien.ginolhac/targets_demos/-/tree/hpc).

``` bash
cd $HOME
git clone --branch hpc https://gitlab.lcsb.uni.lu/aurelien.ginolhac/targets_demos.git
```

- Check you can tart a container inside this newly fetched folder

``` bash
singularity run -H /home/users/${USER}/targets_demos/ --contain \
  /scratch/users/aginolhac/targets_hpc/r-targets.sif ls
```

Here we bind as home only the targets_demos folder (specifying `-H` this new home and `--contain` to not bind the rest of your home) and list its content:

``` bash
LICENSE.md  README.md      _targets_ds_1.R  _targets_ds_2_crew.R  _targets_ds_fun1.R   circles  ds1.Rmd  ds3.Rmd  lines   renv.lock
R           _targets.yaml  _targets_ds_2.R  _targets_ds_3.R       _targets_packages.R  data     ds2.Rmd  img      others  run.R
```


### Using VScode to connect remotely to the HPC

I have followed this tutorial: [R-VScode](https://rolkra.github.io/R-VSCode/) by Roland Krasser



### Useful links

-   [CRAN Archive](https://cran.r-project.org/)
-   [CRAN HPC
    Packages](https://cran.r-project.org/web/views/HighPerformanceComputing.html)
-   [Tidyverse Documentation](https://tidyverse.org/)
-   [4-days tidyverse workshop.uni.lu](https://rworkshop.uni.lu/)
-   [Advanced R programming by Hadley Wickham](https://adv-r.hadley.nz)
