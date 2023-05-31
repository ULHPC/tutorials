#!/bin/bash
set -e

case "$1" in
    sh|bash)
        set -- "$@"
        exec "$@"
    ;;
    sparkMaster)
        shift
        echo "Running Spark Master on `hostname` ${SLURM_PROCID}"
        /opt/spark/sbin/start-master.sh "$@" 1>$HOME/sparkMaster.out 2>&1 &
        status=$?
        if [ $status -ne 0 ]; then
            echo "Failed to start Spark Master: $status"
            exit $status
        fi

        exec tail -f $(ls -Art $HOME/sparkMaster.out | tail -n 1)
    ;;
    sparkWorker)
        shift
        /opt/spark/sbin/start-worker.sh "$@" 1>$HOME/sworker-${SLURM_PROCID}.out 2>&1 &
        status=$?
        if [ $status -ne 0 ]; then
            echo "Failed to start Spark worker: $status"
            exit $status
        fi

        exec tail -f $(ls -Art $HOME/sworker-${SLURM_PROCID}.out | tail -n 1)
    ;;
    sparkHDFSNamenode)
        shift
        echo "Running HDFS Namenode on `hostname` ${SLURM_PROCID}"
        /opt/hadoop/bin/hdfs namenode -format
        echo "Done format"
        /opt/hadoop/bin/hdfs --daemon start namenode "$@" 1>$HOME/hdfsNamenode.out 2>&1 &

        status=$?
        if [ $status -ne 0 ]; then
            echo "Failed to start HDFS Namenode: $status"
            exit $status
        fi

        exec tail -f $(ls -Art $HOME/hdfsNamenode.out | tail -n 1)
    ;;
    sparkHDFSDatanode)
        shift
        echo "Running HDFS datanode on `hostname` ${SLURM_PROCID}"
        /opt/hadoop/bin/hdfs --daemon start datanode "$@" 1>$HOME/hdfsDatanode-${SLURM_PROCID}.out 2>&1 &

        status=$?
        if [ $status -ne 0 ]; then
            echo "Failed to start HDFS datanode: $status"
            exit $status
        fi

        exec tail -f $(ls -Art $HOME/hdfsDatanode-${SLURM_PROCID}.out | tail -n 1)
    ;;
esac

