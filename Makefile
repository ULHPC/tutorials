####################################################################################
# Makefile (configuration file for GNU make - see http://www.gnu.org/software/make/)
# Time-stamp: <Lun 2013-03-11 23:13 svarrette>
#     __  __       _         __ _ _       
#    |  \/  | __ _| | _____ / _(_) | ___  
#    | |\/| |/ _` | |/ / _ \ |_| | |/ _ \
#    | |  | | (_| |   <  __/  _| | |  __/ 
#    |_|  |_|\__,_|_|\_\___|_| |_|_|\___| 
#
# Copyright (c) 2012 Sebastien Varrette <Sebastien.Varrette@uni.lu>
# .             http://varrette.gforge.uni.lu
#
####################################################################################
#
############################## Variables Declarations ##############################
SHELL = /bin/bash

UNAME = $(shell uname)

# Some directories
SUPER_DIR   = $(shell basename `pwd`)

# Git stuff management
GITFLOW      = $(shell which git-flow)
LAST_TAG_COMMIT = $(shell git rev-list --tags --max-count=1)
LAST_TAG = $(shell git describe --tags $(LAST_TAG_COMMIT) )
TAG_PREFIX = "v"
GITFLOW_BR_MASTER=production
GITFLOW_BR_DEVELOP=devel

VERSION  = $(shell [ -f VERSION ] && head VERSION || echo "0.0.1")
# OR try to guess directly from the last git tag
#VERSION    = $(shell  git describe --tags $(LAST_TAG_COMMIT) | sed "s/^$(TAG_PREFIX)//")
MAJOR      = $(shell echo $(VERSION) | sed "s/^\([0-9]*\).*/\1/")
MINOR      = $(shell echo $(VERSION) | sed "s/[0-9]*\.\([0-9]*\).*/\1/")
PATCH      = $(shell echo $(VERSION) | sed "s/[0-9]*\.[0-9]*\.\([0-9]*\).*/\1/")
# total number of commits 		
BUILD      = $(shell git log --oneline | wc -l | sed -e "s/[ \t]*//g")

#REVISION   = $(shell git rev-list $(LAST_TAG).. --count)
#ROOTDIR    = $(shell git rev-parse --show-toplevel)
NEXT_MAJOR_VERSION = $(shell expr $(MAJOR) + 1).0.0-b$(BUILD)
NEXT_MINOR_VERSION = $(MAJOR).$(shell expr $(MINOR) + 1).0-b$(BUILD)
NEXT_PATCH_VERSION = $(MAJOR).$(MINOR).$(shell expr $(PATCH) + 1)-b$(BUILD)

### Main variables

.PHONY: all archive clean help release setup start_bump_major start_bump_minor start_bump_patch test versioninfo 

############################### Now starting rules ################################
# Required rule : what's to be done each time 
all: 

# Test values of variables - for debug purposes  
test:
	@echo "--- Compilation commands --- "
	@echo "GITFLOW      -> '$(GITFLOW)'"
	@echo "--- Directories --- "
	@echo "SUPER_DIR    -> '$(SUPER_DIR)'"
	@echo ""
	@echo "Consider running 'make versioninfo' to get info on git versionning variables"

############################### Archiving ################################
archive: clean
	tar -C ../ -cvzf ../$(SUPER_DIR)-$(VERSION).tar.gz --exclude ".svn" --exclude ".git"  --exclude "*~" --exclude ".DS_Store" $(SUPER_DIR)/

############################### Git Bootstrapping rules ################################
setup:
	git fetch origin
	git branch --set-upstream $(GITFLOW_BR_MASTER) origin/$(GITFLOW_BR_MASTER)
	git config gitflow.branch.master     $(GITFLOW_BR_MASTER)
	git config gitflow.branch.develop    $(GITFLOW_BR_DEVELOP)
	git config gitflow.prefix.feature    feature/
	git config gitflow.prefix.release    release/
	git config gitflow.prefix.hotfix     hotfix/
	git config gitflow.prefix.support    support/
	git config gitflow.prefix.versiontag $(TAG_PREFIX)
	git submodule init
	git submodule update

versioninfo:
	@echo "Current version: $(VERSION)"
	@echo "Last tag: $(LAST_TAG)"
	@echo "$(shell git rev-list $(LAST_TAG).. --count) commit(s) since last tag"
	@echo "Build: $(BUILD) (total number of commits)"
	@echo "next major version: $(NEXT_MAJOR_VERSION)"
	@echo "next minor version: $(NEXT_MINOR_VERSION)"
	@echo "next patch version: $(NEXT_PATCH_VERSION)"

# Git flow management - this should be factorized 
ifeq ($(GITFLOW),)
start_bump_patch start_bump_minor start_bump_major release: 
	@echo "Unable to find git-flow on your system. "
	@echo "See https://github.com/nvie/gitflow for installation details"
else
start_bump_patch: clean
	@echo "Start the patch release of the repository from $(VERSION) to $(NEXT_PATCH_VERSION)"
	git pull origin
	git flow release start $(NEXT_PATCH_VERSION)
	@echo $(NEXT_PATCH_VERSION) > VERSION
	git commit -s -m "Patch bump to version $(NEXT_PATCH_VERSION)" VERSION
	@echo "=> remember to update the version number in $(MAIN_TEX)"
	@echo "=> run 'make release' once you finished the bump"

start_bump_minor: clean
	@echo "Start the minor release of the repository from $(VERSION) to $(NEXT_MINOR_VERSION)"
	git pull origin
	git flow release start $(NEXT_MINOR_VERSION)
	@echo $(NEXT_MINOR_VERSION) > VERSION
	git commit -s -m "Minor bump to version $(NEXT_MINOR_VERSION)" VERSION
	@echo "=> remember to update the version number in $(MAIN_TEX)"
	@echo "=> run 'make release' once you finished the bump"

start_bump_major: clean
	@echo "Start the major release of the repository from $(VERSION) to $(NEXT_MAJOR_VERSION)"
	git pull origin
	git flow release start $(NEXT_MAJOR_VERSION)
	@echo $(NEXT_MAJOR_VERSION) > VERSION
	git commit -s -m "Major bump to version $(NEXT_MAJOR_VERSION)" VERSION
	@echo "=> remember to update the version number in $(MAIN_TEX)"
	@echo "=> run 'make release' once you finished the bump"


release: clean 
	git flow release finish -s $(VERSION)
	git checkout $(GITFLOW_BR_MASTER)
	git push origin
	git checkout $(GITFLOW_BR_DEVELOP)
	git push origin
	git push origin --tags
endif


# Clean option
clean:
	@echo nothing to be cleaned for the moment


# # force recompilation
# force :
# 	@touch $(MAIN_TEX)
# 	@$(MAKE)


# print help message
help :
	@echo '+----------------------------------------------------------------------+'
	@echo '|                        Available Commands                            |'
	@echo '+----------------------------------------------------------------------+'
	@echo '| make setup:   Initiate git-flow for your local copy of the repository|'
	@echo '| make start_bump_{major,minor,patch}: start a new version release with|'
	@echo '|               git-flow at a given level (major, minor or patch bump) |'
	@echo '| make release: Finalize the release using git-flow                    |'
	@echo '+----------------------------------------------------------------------+'

