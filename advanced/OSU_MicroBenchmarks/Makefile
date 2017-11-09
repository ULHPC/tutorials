##################################################################################
# Makefile - Configuration file for GNU make (http://www.gnu.org/software/make/)
# Creation : 07 Nov 2013
# Time-stamp: <Thu 2017-11-09 14:09 svarrette>
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
BENCH_SHORTNAME = "OSU"
BENCH_DESC      = "OSU Micro-Benchmarks"
BENCH_VERSION   = 5.4
BENCH_URL       = "http://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-$(BENCH_VERSION).tar.gz"
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
	$(BUILD_SCRIPT)  $*
	@echo "=> building complete - see build.*"

plot:
	$(MAKE) -C $(PLOT_DIR)/

clean:
	@echo "=> removing build directories"
	rm -rf  build.*
	@echo "=> removing $(BENCH_TARBALL)"
	rm -f $(SRC_DIR)/$(BENCH_TARBALL)
	$(MAKE) -C $(PLOT_DIR)/ $@



ifndef OAR_NODEFILE
run_interactive _run_interactive.impi __run_interactive.openmpi: # __run_interactive.mvapich2:
	@echo "OAR_NODEFILE is not set - you have to be in an interactive job"

run: __run.impi

__run.%:
	ln -sf $(shell basename $(DEFAULT_LAUNCHER_SCRIPT)) $(RUN_DIR)/launcher_osu_$*
	@echo "FIXME: not working"
	@echo oarsub -n "OSU $*" -O OSU_$*-%jobid%.log -E OSU_$*-%jobid%.log -l enclosure=1/nodes=2/core=1 -S "./$(RUN_DIR)/launcher_osu_$* --datadir $(RUN_DIR)/data/`date +%Y-%m-%d`"


else
run __run.impi __run.openmpi: # __run.mvapich2:
	@echo "OAR_NODEFILE is set - you have to be on the access to be able to submit a passive job"

run_interactive: build __run_interactive.impi __run_interactive.openmpi # __run_interactive.mvapich2

__run_interactive.%:
	ln -sf $(shell basename $(DEFAULT_LAUNCHER_SCRIPT)) $(RUN_DIR)/launcher_osu_$*
	./$(RUN_DIR)/launcher_osu_$* --datadir $(RUN_DIR)/data/`date +%Y-%m-%d`

endif
