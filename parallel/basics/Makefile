##################################################################################
# Makefile - Configuration file for GNU make (http://www.gnu.org/software/make/)
# Time-stamp: <Sat 2018-06-09 15:11 svarrette>
#
# Copyright (c) 2013-2018 Sebastien Varrette <Sebastien.Varrette@uni.lu>
#               http://varrette.gforge.uni.lu
#
# Available Commands
# ------------------
# make           : Compile files, binaries are generated in the current directory
# make clean     : Remove backup files (*~) and other generated files
#
############################## Variables Declarations ############################
SHELL = /bin/bash

CC      = gcc
CFLAGS  = -Wall -Wextra -pedantic --std=c99 -O2

#TOP_SRCDIR  = ../..
SRC_DIR     = src
BUILD_DIR   = bin
# RUN_DIR     = runs
# SCRIPT_DIR  = scripts
# PLOT_DIR    = plots

SRC     = $(wildcard $(SRC_DIR)/*.c)
SRC_OMP = $(wildcard $(SRC_DIR)/*openmp*.c)
SRC_MPI = $(wildcard $(SRC_DIR)/*mpi*.c)

TARGET = $(SRC:$(SRC_DIR)/%.c=%)
TARGET_OMP       = $(SRC_OMP:$(SRC_DIR)/%.c=$(BUILD_DIR)/%)
TARGET_OMP_INTEL = $(SRC_OMP:$(SRC_DIR)/%.c=$(BUILD_DIR)/intel_%)

TARGET_OPENMPI  = $(filter-out %openmp, $(addprefix $(BUILD_DIR)/openmpi_,  $(TARGET)))
TARGET_INTEL    = $(filter-out %openmp, $(addprefix $(BUILD_DIR)/intel_,    $(TARGET)))
TARGET_MVAPICH2 = $(filter-out %openmp, $(addprefix $(BUILD_DIR)/mvapich2_, $(TARGET)))

TARGETS = $(TARGET_OMP) $(TARGET_OMP_INTEL) $(TARGET_OPENMPI) $(TARGET_INTEL) $(TARGET_MVAPICH2)

DATARACEBENCH_GIT = https://github.com/LLNL/dataracebench.git
DATARACEBENCH_DIR = $(shell basename $(DATARACEBENCH_GIT) '.git')

############################
.PHONY: all fetch print build
#uncompress configure build clean

all: build

print:
	@echo "SRC     = $(SRC)"
	@echo "SRC_OMP = $(SRC_OMP)"
	@echo "SRC_MPI = $(SRC_MPI)"
	@echo "==============="
	@echo "TARGET  = $(TARGET)"
	@echo "TARGET_OMP       = $(TARGET_OMP)"
	@echo "TARGET_OMP_INTEL = $(TARGET_OMP_INTEL)"
	@echo "TARGET_OPENMPI   = $(TARGET_OPENMPI)"
	@echo "TARGET_INTEL     = $(TARGET_INTEL)"
	@echo "TARGET_MVAPICH2  = $(TARGET_MVAPICH2)"
	@echo "==============="
	@echo "(all) TARGETS = $(TARGETS)"


fetch:
	@if [ ! -d ./$(SRC_DIR)/$(DATARACEBENCH_DIR) ]; then \
	  echo "cloning $(DATARACEBENCH_GIT)"; \
    git clone $(DATARACEBENCH_GIT) ./$(SRC_DIR)/$(DATARACEBENCH_DIR); \
	else \
		echo "=> $(DATARACEBENCH_DIR) has already been cloned in $(SRC_DIR)/"; \
	fi

build: __build.omp __build.openmpi __build.intel __build.mvapich2

##### OpenMP builds
__build.omp: __build.omp.foss __build.omp.intel
__build.omp.foss:  $(TARGET_OMP)
__build.omp.intel: $(TARGET_OMP_INTEL)

$(BUILD_DIR)/%_openmp: $(SRC_DIR)/%_openmp.c
	@echo "==> OpenMP example build (from $<) with the foss toolchain"
	@echo . /etc/profile
	@echo module purge
	@echo "module load toolchain/foss && $(CC) -fopenmp $(CFLAGS) $< -o $@"

$(BUILD_DIR)/intel_%_openmp: $(SRC_DIR)/%_openmp.c
	@echo "==> OpenMP example build (from $<) with the intel toolchain"
	@echo . /etc/profile
	@echo module purge
	@echo "module load toolchain/intel && icc -qopenmp $(CFLAGS) $< -o $@"

##### OpenMPI and hybrid OpenMP+OpenMPI builds
__build.openmpi: $(TARGET_OPENMPI)
$(BUILD_DIR)/openmpi_%: $(SRC_DIR)/%.c
	@echo "==> OpenMPI example build (from $<)"
	@echo . /etc/profile
	@echo module purge
	@echo "module load mpi/OpenMPI && mpicc -fopenmp $(CFLAGS) $< -o $@"


##### Intel MPI and hybrid OpenMP+Intel MPI builds
__build.intel: $(TARGET_INTEL)
$(BUILD_DIR)/intel_%: $(SRC_DIR)/%.c
	@echo "==> Intel MPI example build (from $<)"
	@echo . /etc/profile
	@echo module purge
	@echo "module load toolchain/intel && mpiicc -qopenmp $(CFLAGS) $< -o $@"

##### MVAPICH2 and hybrid OpenMP+MVAPICH2 builds
__build.mvapich2: $(TARGET_MVAPICH2)
$(BUILD_DIR)/mvapich2_%: $(SRC_DIR)/%.c
	@echo "==> MVAPICH2 example build (from $<)"
	@echo . /etc/profile
	@echo module purge
	@echo "module load mpi/MVAPICH2 && mpicc -fopenmp $(CFLAGS) $< -o $@"


clean:
	@echo "=> removing $(BENCH_TARBALL) and $(BENCH_SRCDIR)"
	rm -rf $(SRC_DIR)/$(DATARACEBENCH_DIR)
	rm -f  $(TARGETS)

# #	$(MAKE) -C $(PLOT_DIR)/ $@
# # @echo "=> removing build directories"
# # rm -rf  build.*
