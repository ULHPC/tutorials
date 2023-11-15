#/bin/sh

#experiment() {
#  echo "running ${1} ..."
#  sleep 3
#  echo "${1} finished at $(date)"
#}

#export -f experiment # Make this function available to child processes. Child processes will be launched with parallel

export experiment=${PWD}/experiment.sh
parallel -j 2 $experiment {} ::: a b c d e f
