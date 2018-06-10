##################################################################################
# Makefile - Configuration file for GNU make (http://www.gnu.org/software/make/)
# Time-stamp: <Sun 2018-06-10 14:03 svarrette>
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
# intel compiler flags -- see also
#  https://software.intel.com/en-us/articles/step-by-step-optimizing-with-intel-c-compiler
ICFLAGS = -Wall -xhost -O2

SRC_DIR     = src
BUILD_DIR   = bin


SRC     = $(wildcard $(SRC_DIR)/*.c)
SRC_OMP = $(wildcard $(SRC_DIR)/*openmp*.c)
#SRC_MPI = $(wildcard $(SRC_DIR)/*mpi*.c)

TARGET = $(SRC:$(SRC_DIR)/%.c=%)
TARGET_OMP       = $(SRC_OMP:$(SRC_DIR)/%.c=$(BUILD_DIR)/%)
TARGET_OMP_INTEL = $(SRC_OMP:$(SRC_DIR)/%.c=$(BUILD_DIR)/intel_%)

TARGET_OPENMPI  = $(filter-out %openmp, $(addprefix $(BUILD_DIR)/openmpi_,  $(TARGET)))
TARGET_INTEL    = $(filter-out %openmp, $(addprefix $(BUILD_DIR)/intel_,    $(TARGET)))
TARGET_MVAPICH2 = $(filter-out %openmp, $(addprefix $(BUILD_DIR)/mvapich2_, $(TARGET)))
TARGET_MPI      = $(TARGET_OPENMPI) $(TARGET_INTEL) $(TARGET_MVAPICH2)

TARGETS = $(TARGET_OMP) $(TARGET_OMP_INTEL) $(TARGET_MPI)

DATARACEBENCH_GIT = https://github.com/LLNL/dataracebench.git
DATARACEBENCH_DIR = $(shell basename $(DATARACEBENCH_GIT) '.git')

############################
.PHONY: all fetch print build  omp openmp openmpi intel mvapich2
#uncompress configure build clean

all: build

# setup:


print:
	@echo "SRC     = $(SRC)"
	@echo "SRC_OMP = $(SRC_OMP)"
	@echo "==============="
	@echo "TARGET  = $(TARGET)"
	@echo "TARGET_OMP       = $(TARGET_OMP)"
	@echo "TARGET_OMP_INTEL = $(TARGET_OMP_INTEL)"
	@echo "TARGET_OPENMPI   = $(TARGET_OPENMPI)"
	@echo "TARGET_INTEL     = $(TARGET_INTEL)"
	@echo "TARGET_MVAPICH2  = $(TARGET_MVAPICH2)"
	@echo "TARGET_MPI       = $(TARGET_MPI)"
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
mpi:    $(filter-out %hybrid,  $(TARGET_MPI))
hybrid: $(filter-out %mpi,     $(TARGET_MPI))
omp openmp: __build.omp
openmpi:    __build.openmpi
intel:      __build.omp.intel __build.intel
mvapich2:   __build.mvapich2

##### OpenMP builds
__build.omp: __build.omp.foss __build.omp.intel
__build.omp.foss:  $(TARGET_OMP)
__build.omp.intel: $(TARGET_OMP_INTEL)

$(BUILD_DIR)/%_openmp: $(SRC_DIR)/%_openmp.c
	@echo "==> OpenMP example build (from $<) with the foss toolchain"
	. /etc/profile
	module purge
	module load toolchain/foss && $(CC) -fopenmp $(CFLAGS) $< -o $@

$(BUILD_DIR)/intel_%_openmp: $(SRC_DIR)/%_openmp.c
	@echo "==> OpenMP example build (from $<) with the intel toolchain"
	. /etc/profile
	module purge
	module load toolchain/intel && icc -qopenmp $(ICFLAGS) $< -o $@

##### OpenMPI and hybrid OpenMP+OpenMPI builds
__build.openmpi: $(TARGET_OPENMPI)
$(BUILD_DIR)/openmpi_%: $(SRC_DIR)/%.c
	@echo "==> OpenMPI example build (from $<)"
	. /etc/profile
	module purge
	module load mpi/OpenMPI && mpicc -fopenmp $(CFLAGS) $< -o $@


##### Intel MPI and hybrid OpenMP+Intel MPI builds
__build.intel: $(TARGET_INTEL)
$(BUILD_DIR)/intel_%: $(SRC_DIR)/%.c
	@echo "==> Intel MPI example build (from $<)"
	. /etc/profile
	module purge
	module load toolchain/intel && mpiicc -qopenmp $(ICFLAGS) $< -o $@

##### MVAPICH2 and hybrid OpenMP+MVAPICH2 builds
__build.mvapich2: $(TARGET_MVAPICH2)
$(BUILD_DIR)/mvapich2_%: $(SRC_DIR)/%.c
	@echo "==> MVAPICH2 example build (from $<)"
	. /etc/profile
	module purge
	module load mpi/MVAPICH2 && mpicc -fopenmp $(CFLAGS) $< -o $@

clean:
	@echo "=> removing $(DATARACEBENCH_DIR) and binaries"
	rm -rf $(SRC_DIR)/$(DATARACEBENCH_DIR)
	rm -f  $(TARGETS)
