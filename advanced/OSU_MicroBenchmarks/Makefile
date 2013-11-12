##################################################################################
# Makefile - Configuration file for GNU make (http://www.gnu.org/software/make/)
# Creation : 07 Nov 2013
# Time-stamp: <Mar 2013-11-12 12:00 svarrette>
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

TOP_SRCDIR  = ../..
SRC_DIR     = src
PLOT_DIR    = plots
SAMPLE_PLOT = $(PLOT_DIR)/benchmark_OSU_MicroBenchs_2H_latency.pdf $(PLOT_DIR)/benchmark_OSU_MicroBenchs_2H_bandwidth.pdf


### Stuff to download
OSU_MB_URL       = "http://mvapich.cse.ohio-state.edu/benchmarks/osu-micro-benchmarks-4.2.tar.gz"
OSU_MB_TARBALL   = $(shell basename $(OSU_MB_URL) )  
OSU_MB_SRCDIR    = $(SRC_DIR)/$(shell basename $(OSU_MB_TARBALL) .tar.gz)


############################
.PHONY: all fetch clean plot 

all: fetch plot

fetch:
	@if [ ! -f ./$(SRC_DIR)/$(OSU_MB_TARBALL) ]; then \
		echo "=> downloading HPL archive  $(OSU_MB_TARBALL)"; \
		wget -P $(SRC_DIR) $(OSU_MB_URL); \
	else \
		echo "=> OSU Micro-benchmarks has already been downloaded in $(SRC_DIR)/"; \
	fi

build: fetch
	@echo "TODO: define the building approach"

run_interactive:
	@echo "TODO: define action to run the code interactively"

run:
	@echo "TODO: define action to run the code, most probably oarsub -S ..."

plot:
	$(MAKE) -C $(PLOT_DIR)/
	@echo "=> You probably want now to see the comparaison plot: \n\t\t$(SAMPLE_PLOT)"

clean: 
	@echo "=> removing $(OSU_MB_TARBALL)"
	rm -f $(SRC_DIR)/$(OSU_MB_TARBALL)
	$(MAKE) -C $(PLOT_DIR)/ $@
