#!/bin/bash
# Copyright (c) 2016, Intel Corporation

mic_exe="./hybrid_demo_mic"
host_exe="./heart_demo"
app_args="-m ../vs -s ../setup_mid.txt -v -i  -t 500"
mic_extra='-genv LD_LIBRARY_PATH "/nfs/inn/proj/mpi/pdsd/opt/EM64T-LIN/compilers/intel/composer_xe_2013.3.163/compiler/lib/mic/:$LD_LIBRARY_PATH'
export I_MPI_MIC=enable
# export OMP_NUM_THREADS=61
mpi_mic_np=1

mpiexec.hydra --print-rank-map \
      -n $mpi_mic_np  -host nnlmpimic11-mic0  $mic_extra $mic_exe $app_args \
    : -n $mpi_mic_np  -host nnlmpimic13-mic0  $mic_extra $mic_exe $app_args \
    : -n 1 -host nnlmpimic11 $host_exe $app_args

# mpirun --print-rank-map \
#       -n 1  -host nnlmpimic11-mic0  $mic_extra $mic_exe $app_args \
#     : -n 1  -host nnlmpimic12-mic0  $mic_extra $mic_exe $app_args \
#     : -n 1  -host nnlmpimic13-mic0  $mic_extra $mic_exe $app_args \
#     : -n 1 -host nnlmpimic11 $host_exe $app_args


# mpirun --print-rank-map -n 4 -ppn 2   -host nnlmpimic12,nnlmpimic13 ./heart_demo -m ../vs -s ../setup_mid.txt -v -i  -t 500 : -n 1 -host nnlmpimic11 ./heart_demo -m ../vs -s ../setup_mid.txt -v -i -t 500
