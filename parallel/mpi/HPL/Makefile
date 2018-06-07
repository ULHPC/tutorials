##################################################################################
# Makefile - Configuration file for GNU make (http://www.gnu.org/software/make/)
# Creation : 07 Nov 2013
# Time-stamp: <Mon 2017-06-12 13:14 svarrette>
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
BENCH_SHORTNAME = "HPL"
BENCH_DESC      = "HPL: High-Performance Linpack Benchmark"
BENCH_VERSION   = 2.2
BENCH_URL       = "http://www.netlib.org/benchmark/hpl/hpl-$(BENCH_VERSION).tar.gz"
BENCH_TARBALL   = $(shell basename $(BENCH_URL) )
BENCH_SRCDIR    = $(SRC_DIR)/$(shell basename $(BENCH_TARBALL) .tar.gz)

#BUILD_SCRIPT = ./$(SCRIPT_DIR)/build.$(BENCH_SHORTNAME)

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
		echo "=> $(BENCH_DESC) has already been downloaded in $(SRC_DIR)/"; \
	fi

uncompress: fetch
	@if [ ! -f $(BENCH_SRCDIR)/configure ]; then \
		echo "=> uncompressing the archive $(BENCH_TARBALL)"; \
		tar -C $(SRC_DIR) -xvf $(SRC_DIR)/$(BENCH_TARBALL); \
	else \
		echo "=> the archive $(BENCH_TARBALL) has already been uncompressed"; \
	fi


# TOP_SRCDIR  = ../..
# TARBALL_DIR = tarballs
# SRC_DIR     = src
# PLOT_DIR    = plots
# SAMPLE_PLOT = $(PLOT_DIR)/benchmark_HPL_2H.pdf


# ### Stuff to download
# HPL_URL       = "http://www.netlib.org/benchmark/hpl/hpl-2.1.tar.gz"
# HPL_TARBALL   = $(shell basename $(HPL_URL) )
# HPL_SRCDIR    = $(SRC_DIR)/$(shell basename $(HPL_TARBALL) .tar.gz)

# GOTOBLAS2_URL     = "http://www.tacc.utexas.edu/documents/13601/b58aeb8c-9d8d-4ec2-b5f1-5a5843b4d47b"
# GOTOBLAS2_TARBALL = GotoBLAS2-1.13.tar.gz
# GOTOBLAS2_SRCDIR  = $(SRC_DIR)/$(shell basename $(GOTOBLAS2_TARBALL) .tar.gz)

# ATLAS_URL     = 'http://downloads.sourceforge.net/project/math-atlas/Developer%20%28unstable%29/3.11.17/atlas3.11.17.tar.bz2?r=http%3A%2F%2Fsourceforge.net%2Fprojects%2Fmath-atlas%2Ffiles%2FStable%2F&ts=1383822103&use_mirror=freefr'
# ATLAS_TARBALL = atlas3.11.17.tar.bz2
# ATLAS_SRCDIR  = $(SRC_DIR)/$(shell basename $(GOTOBLAS2_TARBALL) .tar.bz2)






############################
# .PHONY: all fetch clean plot

# all: fetch plot

# fetch:
# 	@if [ ! -f ./$(TARBALL_DIR)/$(HPL_TARBALL) ]; then \
# 		echo "=> downloading HPL archive  $(HPL_TARBALL)"; \
# 		wget -P $(TARBALL_DIR) $(HPL_URL); \
# 	else \
# 		echo "=> HPL has already been downloaded in $(TARBALL_DIR)/"; \
# 	fi
# 	@if [ ! -f ./$(TARBALL_DIR)/$(GOTOBLAS2_TARBALL) ]; then \
# 		echo "=> downloading GotoBLAS2 library archive $(GOTOBLAS2_TARBALL)"; \
# 		wget -O $(TARBALL_DIR)/$(GOTOBLAS2_TARBALL)  $(GOTOBLAS2_URL); \
# 	else \
# 		echo "=> GotoBLAS2 has already been downloaded in $(TARBALL_DIR)/"; \
# 	fi
# 	@if [ ! -f ./$(TARBALL_DIR)/$(ATLAS_TARBALL) ]; then \
# 		echo "=> downloading ATLAS (Automatically Tuned Linear Algebra Soft) archive $(ATLAS_TARBALL)"; \
# 		wget -O $(TARBALL_DIR)/$(ATLAS_TARBALL) $(ATLAS_URL); \
# 	else \
# 		echo "=> ATLAS has already been downloaded in $(TARBALL_DIR)/"; \
# 	fi

# build: fetch
# 	@echo "TODO: define the building approach"

# run_interactive:
# 	@echo "TODO: define action to run the code interactively"

# run:
# 	@echo "TODO: define action to run the code, most probably oarsub -S ..."

# plot:
# 	$(MAKE) -C $(PLOT_DIR)/
# 	@echo "=> You probably want now to see the comparaison plot: \n\t\t$(SAMPLE_PLOT)"

# clean:
# 	@for f in $(HPL_TARBALL) $(GOTOBLAS2_TARBALL) $(ATLAS_TARBALL); do \
# 		echo "=> removing $$f"; \
# 		rm -f  $(TARBALL_DIR)/$$f; \
# 	done
# 	$(MAKE) -C $(PLOT_DIR)/ $@
