#!/bin/sh

# Argument 1 is the input file, argument 2 is the output file
wc -w ${1} >> ${2}

