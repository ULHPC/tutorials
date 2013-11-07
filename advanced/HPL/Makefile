##################################################################################
# Makefile - Configuration file for GNU make (http://www.gnu.org/software/make/)
# Creation : 07 Nov 2013
# Time-stamp: <Thu 2013-11-07 11:56 svarrette>
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
TARBALL_DIR = tarballs

### Stuff to download
HPL_URL       = "http://www.netlib.org/benchmark/hpl/hpl-2.1.tar.gz"
GOTOBLAS2_URL = "http://www.tacc.utexas.edu/documents/13601/b58aeb8c-9d8d-4ec2-b5f1-5a5843b4d47b"

HPL_TARBALL       = $(shell basename $(HPL_URL) )  
GOTOBLAS2_TARBALL = GotoBLAS2-1.13.tar.gz


############################
.PHONY: all bootstrap clean

all:
	@echo "=> no default task"

bootstrap:
	@if [ ! -f ./$(TARBALL_DIR)/$(HPL_TARBALL) ]; then \
		echo "=> downloading HPL $(HPL_TARBALL)"; \
		wget -P $(TARBALL_DIR) $(HPL_URL); \
	else \
		echo "=> HPL has already been downloaded in $(TARBALL_DIR)/"; \
	fi
	@if [ ! -f ./$(TARBALL_DIR)/$(GOTOBLAS2_TARBALL) ]; then \
		echo "=> downloading GotoBLAS2 library $(GOTOBLAS2_TARBALL)"; \
		wget -P $(TARBALL_DIR) $(GOTOBLAS2_URL); \
		mv ./$(TARBALL_DIR)/`basename $(GOTOBLAS2_URL)` ./$(TARBALL_DIR)/$(GOTOBLAS2_TARBALL); \
	else \
		echo "=> GotoBLAS2 has already been downloaded in $(TARBALL_DIR)/"; \
	fi


clean: 
	rm -f $(TARBALL_DIR)/$(HPL_TARBALL)
