##################################################################################
# Makefile - Configuration file for GNU make (http://www.gnu.org/software/make/)
# Creation : 07 Nov 2013
# Time-stamp: <Wed 2021-11-17 22:59 svarrette>
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
BENCH_SHORTNAME = OSU
BENCH_DESC      = "OSU Micro-Benchmarks"
BENCH_VERSION   = 5.8
BENCH_URL       = "http://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-$(BENCH_VERSION).tgz"
BENCH_TARBALL   = $(shell basename $(BENCH_URL) )
BENCH_SRCDIR    = $(SRC_DIR)/$(shell basename $(BENCH_TARBALL) .tar.gz)

BUILD_SCRIPT = ./$(SCRIPT_DIR)/build.$(BENCH_SHORTNAME)

# DEFAULT_LAUNCHER_SCRIPT = ./$(RUN_DIR)/default_mpi_launcher.sh
# LAUNCHER_CONF    = $(wildcard $(RUN_DIR)/*.default.conf)
# LAUNCHER_SCRIPTS = $(LAUNCHER_CONF:.default.conf=)

############################
.PHONY: all fetch clean plot

all: fetch plot

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

build: uncompress __build.intel __build.openmpi # __build.mvapich2

__build.%:
	@echo "=> building $(BENCH_DESC) with $*"
	$(BUILD_SCRIPT) --srcdir $(PWD)/src/osu-micro-benchmarks-$(BENCH_VERSION)  $*
	@echo "=> building complete - see build.*"

plot:
		$(MAKE) -C $(PLOT_DIR)/ \

clean:
	@echo "=> removing build directories"
	rm -rf  build.*
	@echo "=> removing $(BENCH_TARBALL)"
	rm -f $(SRC_DIR)/$(BENCH_TARBALL)
#	$(MAKE) -C $(PLOT_DIR)/ $@
