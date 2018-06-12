####################################################################################
# Makefile (configuration file for GNU make - see http://www.gnu.org/software/make/)
# Time-stamp: <Sat 2017-05-13 22:50 svarrette>
#
#     __  __       _         __ _ _
#    |  \/  | __ _| | _____ / _(_) | ___
#    | |\/| |/ _` | |/ / _ \ |_| | |/ _ \
#    | |  | | (_| |   <  __/  _| | |  __/
#    |_|  |_|\__,_|_|\_\___|_| |_|_|\___|
#
# Copyright (c) 2011 Sebastien Varrette <Sebastien.Varrette@uni.lu>
# .             http://varrette.gforge.uni.lu
#
####################################################################################
#
############################## Variables Declarations ##############################

GNUPLOTFILES   = $(wildcard *.gnuplot)
GNUPLOT_DATA   = $(wildcard data/*.dat) $(wildcard *.dat)
GNUPLOT_OUTPUT = $(shell grep '^set output' *.gnuplot | awk '{ print $$3 }')
OUTPUT_EXT     = $(shell echo $(suffix $(GNUPLOT_OUTPUT)) | tr ' ' '\n' | sort | uniq)
TARGETS  = plot

# Makefile custom hooks
MAKEFILE_BEFORE = .Makefile.before
MAKEFILE_AFTER  = .Makefile.after

ifneq (,$(wildcard $(MAKEFILE_BEFORE)))
include $(MAKEFILE_BEFORE)
endif

all: $(TARGETS)

plot: $(GNUPLOTFILES) $(GNUPLOT_DATA)
	@for plotscript in $(GNUPLOTFILES); do \
		echo "=> processing the GNUPlot script '$$plotscript'"; \
		gnuplot $$plotscript; \
	done

print:
	@echo "GNUPLOTFILES   = $(GNUPLOTFILES)"
	@echo "GNUPLOT_DATA   = $(GNUPLOT_DATA)"
	@echo "GNUPLOT_OUTPUT = $(GNUPLOT_OUTPUT)"
	@echo "OUTPUT_EXT     = $(OUTPUT_EXT)"

%.pdf: %.eps
	epstopdf $<

%.jpg: %.pdf
	convert -verbose -density 150 -trim $< -quality 100 -flatten -sharpen 0x1.0 $@

%.png: %.pdf
	convert -density 400 -depth 8 -quality 100 $< $@

clean:
	@if [ "$(GNUPLOT_OUTPUT)" != "" ]; then \
		echo "=> removing $(GNUPLOT_OUTPUT)"; \
		rm -f $(GNUPLOT_OUTPUT); \
	fi


ifneq (,$(wildcard $(MAKEFILE_AFTER)))
include $(MAKEFILE_AFTER)
endif


#%:
#	$(MAKE) -C ../ $@
