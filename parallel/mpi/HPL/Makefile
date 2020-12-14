##################################################################################
# Makefile - Configuration file for GNU make (http://www.gnu.org/software/make/)
# Creation : 07 Nov 2013
# Time-stamp: <Tue 2019-11-19 23:58 svarrette>
#
# Copyright (c) 2013 Sebastien Varrette <Sebastien.Varrette@uni.lu>
#               http://varrette.gforge.uni.lu
#
# Available Commands
# ------------------
# make           : Compile files, binaries are generated in the current directory
# make force     : Force the complete re-compilation, even if not needed
# make clean     : Remove backup files (*~) and other generated files
#
############################## Variables Declarations ############################
SHELL = /bin/bash

TOP_SRCDIR  = ../..
SRC_DIR     = src
RUN_DIR     = runs
SCRIPT_DIR  = scripts
PLOT_DIR    = plots

### Benchmark info
BENCH_SHORTNAME = HPL
BENCH_DESC      = "HPL: High-Performance Linpack Benchmark"
BENCH_VERSION   = 2.3
BENCH_URL       = "http://www.netlib.org/benchmark/hpl/hpl-$(BENCH_VERSION).tar.gz"
BENCH_TARBALL   = $(shell basename $(BENCH_URL) )
BENCH_SRCDIR    = $(SRC_DIR)/$(shell basename $(BENCH_TARBALL) .tar.gz)

BUILD_SCRIPT = ./$(SCRIPT_DIR)/build.$(BENCH_SHORTNAME)

############################
.PHONY: all fetch clean plot

all: fetch

setup:
	$(MAKE) -C $(TOP_SRCDIR) setup

fetch:
	@if [ ! -f ./$(SRC_DIR)/$(BENCH_TARBALL) ]; then \
		echo "=> downloading benchmark archive  $(BENCH_TARBALL)"; \
		wget --no-check-certificate -P $(SRC_DIR) $(BENCH_URL); \
	else \
		echo "=> $(BENCH_TARBALL) has already been downloaded in $(SRC_DIR)/"; \
	fi

uncompress: fetch
	@if [ ! -f $(BENCH_SRCDIR)/configure ]; then \
		echo "=> uncompressing the archive $(BENCH_TARBALL)"; \
		tar -C $(SRC_DIR) -xvf $(SRC_DIR)/$(BENCH_TARBALL); \
	else \
		echo "=> the archive $(BENCH_TARBALL) has already been uncompressed"; \
	fi

build: uncompress __build.intel #__build.openmpi  __build.mvapich2

__build.%:
	@echo "=> building $(BENCH_DESC) with $*"
	$(BUILD_SCRIPT) -t $(PWD) $*
	@echo "=> building complete - see bin/"


clean:
	@echo "=> removing $(BENCH_TARBALL) and $(BENCH_SRCDIR)"
	rm -f $(SRC_DIR)/$(BENCH_TARBALL)
	rm -rf $(BENCH_SRCDIR)
	rm -f bin/xhpl*

# distclean: clean
# 	rm -f bin/xhpl*

# GOTOBLAS2_URL     = "http://www.tacc.utexas.edu/documents/13601/b58aeb8c-9d8d-4ec2-b5f1-5a5843b4d47b"
# GOTOBLAS2_TARBALL = GotoBLAS2-1.13.tar.gz
# GOTOBLAS2_SRCDIR  = $(SRC_DIR)/$(shell basename $(GOTOBLAS2_TARBALL) .tar.gz)

# ATLAS_URL     = 'http://downloads.sourceforge.net/project/math-atlas/Developer%20%28unstable%29/3.11.17/atlas3.11.17.tar.bz2?r=http%3A%2F%2Fsourceforge.net%2Fprojects%2Fmath-atlas%2Ffiles%2FStable%2F&ts=1383822103&use_mirror=freefr'
# ATLAS_TARBALL = atlas3.11.17.tar.bz2
# ATLAS_SRCDIR  = $(SRC_DIR)/$(shell basename $(GOTOBLAS2_TARBALL) .tar.bz2)
