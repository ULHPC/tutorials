##################################################################################
# Makefile - Configuration file for GNU make (http://www.gnu.org/software/make/)
# Creation : 07 Nov 2013
# Time-stamp: <Jeu 2013-11-14 16:36 svarrette>
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

SAMPLE_PLOT = $(PLOT_DIR)/benchmark_OSU_MicroBenchs_2H_latency.pdf $(PLOT_DIR)/benchmark_OSU_MicroBenchs_2H_bandwidth.pdf

### Stuff to download
OSU_MB_URL       = "http://mvapich.cse.ohio-state.edu/benchmarks/osu-micro-benchmarks-4.2.tar.gz"
OSU_MB_TARBALL   = $(shell basename $(OSU_MB_URL) )  
OSU_MB_SRCDIR    = $(SRC_DIR)/$(shell basename $(OSU_MB_TARBALL) .tar.gz)

OSU_PATCH_FILE   = $(shell git ls-files *.patch)
OSU_PATCH_DIR    = $(shell dirname $(OSU_PATCH_FILE))
OSU_PATCH_REJ_FILE = $(OSU_PATCH_DIR)/$(shell basename $(OSU_PATCH_FILE) .patch).rej

BUILD_SCRIPT = ./$(SCRIPT_DIR)/build.osu

DEFAULT_LAUNCHER_SCRIPT = ./$(RUN_DIR)/default_mpi_launcher.sh
LAUNCHER_CONF    = $(wildcard $(RUN_DIR)/*.default.conf)
LAUNCHER_SCRIPTS = $(LAUNCHER_CONF:.default.conf=)

############################
.PHONY: all fetch clean plot 

all: fetch plot

setup: 
	$(MAKE) -C $(TOP_SRCDIR) setup

fetch:
	@if [ ! -f ./$(SRC_DIR)/$(OSU_MB_TARBALL) ]; then \
		echo "=> downloading HPL archive  $(OSU_MB_TARBALL)"; \
		wget -P $(SRC_DIR) $(OSU_MB_URL); \
	else \
		echo "=> OSU Micro-benchmarks has already been downloaded in $(SRC_DIR)/"; \
	fi

uncompress: fetch
	@if [ ! -f $(OSU_MB_SRCDIR)/configure ]; then \
		echo "=> uncompressing the archive $(OSU_MB_TARBALL)"; \
		tar -C $(SRC_DIR) -xvf $(SRC_DIR)/$(OSU_MB_TARBALL); \
	else \
		echo "=> the archive $(OSU_MB_TARBALL) has already been uncompressed"; \
	fi

patch : uncompress
	@if [ ! -f "./$(OSU_PATCH_REJ_FILE)" ]; then \
		echo "=> apply the patch for 4.2"; \
		patch --forward -d $(shell dirname $(OSU_PATCH_FILE)) -p0 < $(OSU_PATCH_FILE); \
		echo "=> running back automake to commit the patch"; \
		cd $(OSU_MB_SRCDIR) && pwd && autoreconf && automake && cd -; \
	else \
		echo "=> the patch has already been applied"; \
	fi

build: patch __build.impi __build.openmpi __build.mvapich2

__build.%:
	@echo "=> building OSU Micro-benchmarks with $*"
	$(BUILD_SCRIPT)  $*
	@echo "=> building complete - see $(OSU_MB_SRCDIR)/build.*"





ifndef OAR_NODEFILE
run_interactive _run_interactive.impi __run_interactive.openmpi __run_interactive.mvapich2: 
	@echo "OAR_NODEFILE is not set - you have to be in an interactive job"

run: __run.impi

__run.%:
	ln -sf $(shell basename $(DEFAULT_LAUNCHER_SCRIPT)) $(RUN_DIR)/launcher_osu_$*
	@echo "FIXME: not working" 
	@echo oarsub -n "OSU $*" -O OSU_$*-%jobid%.log -E OSU_$*-%jobid%.log -l enclosure=1/nodes=2/core=1 -S "./$(RUN_DIR)/launcher_osu_$* --datadir $(RUN_DIR)/data/`date +%Y-%m-%d`"


else
run __run.impi __run.openmpi __run.mvapich2: 
	@echo "OAR_NODEFILE is set - you have to be on the access to be able to submit a passive job"

run_interactive: build __run_interactive.impi __run_interactive.openmpi __run_interactive.mvapich2

__run_interactive.%:
	ln -sf $(shell basename $(DEFAULT_LAUNCHER_SCRIPT)) $(RUN_DIR)/launcher_osu_$*
	./$(RUN_DIR)/launcher_osu_$* --datadir $(RUN_DIR)/data/`date +%Y-%m-%d`

endif


plot:
	$(MAKE) -C $(PLOT_DIR)/
	@echo "=> You probably want now to see the comparaison plot: \n\t\t$(SAMPLE_PLOT)"

clean: 
	@if [ -f ./$(SRC_DIR)/$(OSU_MB_TARBALL) ]; then \
		echo "=> removing content of the archive file"; \
		tar -tf $(SRC_DIR)/$(OSU_MB_TARBALL) | tail -n +2 | sed 's/^osu/\.\/src\/osu/' | xargs rm -f; \
		git checkout $(OSU_MB_SRCDIR); \
	else echo "no file ./$(SRC_DIR)/$(OSU_MB_TARBALL)"; \
	fi
	rm -rf $(OSU_MB_SRCDIR)/autom4te.cache/ $(OSU_MB_SRCDIR)/build.* 
	@echo "=> removing $(OSU_MB_TARBALL)"
	rm -f $(SRC_DIR)/$(OSU_MB_TARBALL)
	@echo "=> removing launchers scripts"
	rm -f $(LAUNCHER_SCRIPTS)
	$(MAKE) -C $(PLOT_DIR)/ $@

test:
	@echo "OSU_PATCH_FILE     = $(OSU_PATCH_FILE)"
	@echo "OSU_PATCH_DIR      = $(OSU_PATCH_DIR)"
	@echo "OSU_PATCH_REJ_FILE = $(OSU_PATCH_REJ_FILE)"
	@echo "LAUNCHER_SCRIPTS   = $(LAUNCHER_SCRIPTS)"

