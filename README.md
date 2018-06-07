[![By ULHPC](https://img.shields.io/badge/by-ULHPC-blue.svg)](https://hpc.uni.lu) [![Licence](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](http://www.gnu.org/licenses/gpl-3.0.html) [![GitHub issues](https://img.shields.io/github/issues/ULHPC/tutorials.svg)](https://github.com/ULHPC/tutorials/issues/) [![Github](https://img.shields.io/badge/sources-github-green.svg)](https://github.com/ULHPC/tutorials/) [![Documentation Status](http://readthedocs.org/projects/ulhpc-tutorials/badge/?version=latest)](http://ulhpc-tutorials.readthedocs.io) [![GitHub forks](https://img.shields.io/github/stars/ULHPC/tutorials.svg?style=social&label=Star)](https://github.com/ULHPC/tutorials)

        _   _ _       _   _ ____   ____   _____      _             _       _
       | | | | |     | | | |  _ \ / ___| |_   _|   _| |_ ___  _ __(_) __ _| |___
       | | | | |     | |_| | |_) | |       | || | | | __/ _ \| '__| |/ _` | / __|
       | |_| | |___  |  _  |  __/| |___    | || |_| | || (_) | |  | | (_| | \__ \
        \___/|_____| |_| |_|_|    \____|   |_| \__,_|\__\___/|_|  |_|\__,_|_|___/

               Copyright (c) 2013-2018 UL HPC Team <hpc-sysadmins@uni.lu>

This repository holds a set of tutorials to help the users of the [UL HPC](https://hpc.uni.lu) platform to better understand or simply use our platform.

* [**Reference online version**](http://ulhpc-tutorials.readthedocs.io)
* The list of the proposed tutorials is continuously evolving and are used on a regular basis during the [UL HPC School](http://hpc.uni.lu/hpc-school/) we organise twice a year at the University of Luxembourg.
    - You can find the up-to-date list of tutorials [on this page](docs/README.md)

## Installation / Repository Setup

First of all, ensure you have installed the [Pre-requisites / Preliminary software](docs/setup/preliminaries.md) and followed the corresponding configuration [instructions](docs/setup/preliminaries.md).

Then reference instructions for setting up your working copy of this repository can be found in [`docs/setup/install.md`](docs/setup/install.md).

In short:

```bash
$> mkdir -p ~/git/github.com/ULHPC
$> cd ~/git/github.com/ULHPC
$> git clone https://github.com/ULHPC/tutorials.git
$> cd tutorials
$> make setup
```
## Tutorial Slides and Instructions

The latest version of all the proposed tutorials is available online:

<http://ulhpc-tutorials.rtfd.io>

For each tutorial, a PDF copy of the slides are provided (as `slides.pdf` in the corresponding sub-directories).

A [List of the proposed tutorials](docs/README.md) is summarized in [`docs/README.md`](docs/README.md).

## List of contributors

See [`docs/contacts.md`](docs/contacts.md)

In the advent where you want to contribute yourself to these tutorials, do not hesitate! See below for instructions.

## Issues / Feature request

You can submit bug / issues / feature request using the [`ULHPC/tutorials` Tracker](https://github.com/ULHPC/tutorials/issues).

## Developments / Contributing to the code

If you want to contribute to the code, you shall be aware of the way this repository is organized.
These elements are detailed in [`docs/contributing.md`](docs/contributing.md).

You are more than welcome to contribute to its development by [sending a pull request](https://help.github.com/articles/using-pull-requests).

## Online Documentation

[Read the Docs](https://readthedocs.org/) aka RTFD hosts documentation for the open source community and the [ULHPC/sysadmins](https://github.com/ULHPC/tutorials) has its documentation (see the `docs/` directly) hosted on [readthedocs](http://ulhpc-tutorials.rtfd.org).

See [`docs/rtfd.md`](docs/rtfd.md) for more details.

## Licence

Unless otherwise specified, this project and the sources proposed within this repository are released under the terms of the [GPL-3.0](LICENCE) licence.

[![Licence](https://www.gnu.org/graphics/gplv3-88x31.png)](LICENSE)
