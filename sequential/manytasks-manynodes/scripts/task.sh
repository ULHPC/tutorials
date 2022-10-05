#!/usr/bin/bash -l

# $PARALLEL_SEQ is a special variable from GNU parallel. It gives the
# number of the job in the sequence.
#
# Here we print the sleep time, host name, and the date and time.
echo "Task ${PARALLEL_SEQ}  started on host:$(hostname) date:$(date)"
echo "Stress test on 1 core"
${HOME}/.local/bin/stress-ng --cpu 1 --timeout 60s --metrics-brief
