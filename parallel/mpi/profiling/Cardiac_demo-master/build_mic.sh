#!/bin/bash
#Copyright (c) 2016, Intel Corporation

mpiicpc  ../heart_demo.cpp ../luo_rudy_1991.cpp ../rcm.cpp ../mesh.cpp -g -o heart_demo_mic -O3 -mmic -std=c++11 -fopenmp
