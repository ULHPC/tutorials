#!/usr/bin/bash -l
#SBATCH --job-name=SparkHDFS
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=16
#SBATCH --time=0-00:59:00
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --mail-user=first.lastname@uni.lu
#SBATCH --mail-type=BEGIN,END

module load tools/Singularity

hostName="`hostname`"
echo "hostname=$hostName"

#save it for future job refs
myhostname="`hostname`"
rm -f coordinatorNode
touch  coordinatorNode
cat > coordinatorNode << EOF
$myhostname
EOF

###

# Ensure that loging and work directories exist
mkdir -p ${HOME}/sparkhdfs/hadoop/logs
mkdir -p ${HOME}/sparkhdfs/hadoop/etc/hadoop
mkdir -p ${HOME}/sparkhdfs/spark/logs
mkdir -p ${HOME}/sparkhdfs/spark/conf
mkdir -p ${HOME}/sparkhdfs/spark/work

#create Spark configs
SPARK_CONF=${HOME}/sparkhdfs/spark/conf/spark-defaults.conf
cat > ${SPARK_CONF} << EOF

# Master settings
spark.master spark://$hostName:7078

# Memory settings
spark.driver.memory 2g
spark.executor.memory 12g

# Cores settings
spark.executor.cores 8
spark.cores.max 16

# Network settings
spark.driver.host $hostName

# Other settings
spark.logConf true

EOF

SPARK_ENVSH=${HOME}/sparkhdfs/spark/conf/spark-env.sh
cat > ${SPARK_ENVSH} << EOF
#!/usr/bin/env bash

SPARK_MASTER_HOST="$hostName"
SPARK_MASTER_PORT="7078"
SPARK_HOME="/opt/spark"
HADOOP_HOME="/opt/hadoop"

EOF

SPARK_L4J=${HOME}/sparkhdfs/spark/conf/log4j.properties
cat > ${SPARK_L4J} << EOF
# Set everything to be logged to the console
log4j.rootCategory=DEBUG, console
log4j.appender.console=org.apache.log4j.ConsoleAppender
log4j.appender.console.target=System.err
log4j.appender.console.layout=org.apache.log4j.PatternLayout
log4j.appender.console.layout.ConversionPattern=%d{yy/MM/dd HH:mm:ss} %p %c{1}: %m%n

# Settings to quiet third party logs that are too verbose
log4j.logger.org.spark_project.jetty=ERROR
log4j.logger.org.spark_project.jetty.util.component.AbstractLifeCycle=ERROR
log4j.logger.org.apache.spark.repl.SparkIMain$exprTyper=INFO
log4j.logger.org.apache.spark.repl.SparkILoop$SparkILoopInterpreter=INFO
EOF

### create HDFS config
HDFS_SITE=${HOME}/sparkhdfs/hadoop/etc/hadoop/hdfs-site.xml
cat > ${HDFS_SITE} << EOF
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>

<configuration>
  <property>
    <name>dfs.replication</name>
    <value>1</value>
  </property>

  <property>
    <name>dfs.namenode.name.dir</name>
    <value>/tmp/hadoop/hdfs/name</value>
  </property>

  <property>
   <name>dfs.datanode.data.dir</name>
   <value>/tmp/hadoop/hdfs/data</value>
  </property>

</configuration>

EOF

HDFS_CORESITE=${HOME}/sparkhdfs/hadoop/etc/hadoop/core-site.xml
cat > ${HDFS_CORESITE} << EOF
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>

<configuration>
  <property>
  <name>fs.defaultFS</name>
  <value>hdfs://$hostName:9000</value>
  </property>

</configuration>

EOF

###

# Create a launcher script for SparkMaster and hdfsNamenode
#Once started, the Spark master will print out a spark://HOST:PORT to be used for submitting jobs

SPARKM_LAUNCHER=${HOME}/sparkhdfs/spark-start-master-${SLURM_JOBID}.sh
echo " - create SparkMaster and hdfsNamenode launcher script '${SPARKM_LAUNCHER}'"
cat << 'EOF' > ${SPARKM_LAUNCHER}
#!/bin/bash

echo "I am ${SLURM_PROCID} running on:"
hostname

#we are going to share an instance for Spark master and HDFS namenode
singularity instance start --bind $HOME/sparkhdfs/hadoop/logs:/opt/hadoop/logs,$HOME/sparkhdfs/hadoop/etc/hadoop:/opt/hadoop/etc/hadoop,$HOME/sparkhdfs/spark/conf:/opt/spark/conf,$HOME/sparkhdfs/spark/logs:/opt/spark/logs,$HOME/sparkhdfs/spark/work:/opt/spark/work \
 sparkhdfs.sif shinst

singularity run --bind $HOME/sparkhdfs/hadoop/logs:/opt/hadoop/logs,$HOME/sparkhdfs/hadoop/etc/hadoop:/opt/hadoop/etc/hadoop instance://shinst \
  sparkHDFSNamenode 2>&1 &

singularity run --bind $HOME/sparkhdfs/spark/conf:/opt/spark/conf,$HOME/sparkhdfs/spark/logs:/opt/spark/logs,$HOME/sparkhdfs/spark/work:/opt/spark/work instance://shinst \
  sparkMaster

#the following example works for running without instance only the Spark Master
#singularity run --bind $HOME/sparkhdfs/spark/conf:/opt/spark/conf,$HOME/sparkhdfs/spark/logs:/opt/spark/logs,$HOME/sparkhdfs/spark/work:/opt/spark/work sparkhdfs.sif \
# sparkMaster

EOF
chmod +x ${SPARKM_LAUNCHER}

srun --exclusive -N 1 -n 1 -c 16 --ntasks-per-node=1 -l -o $HOME/sparkhdfs/SparkMaster-`hostname`.out \
 ${SPARKM_LAUNCHER} &

export SPARKMASTER="spark://$hostName:7078"

echo "Starting Spark workers and HDFS datanodes"

SPARK_LAUNCHER=${HOME}/sparkhdfs/spark-start-workers-${SLURM_JOBID}.sh
echo " - create Spark workers and HDFS datanodes launcher script '${SPARK_LAUNCHER}'"
cat << 'EOF' > ${SPARK_LAUNCHER}
#!/bin/bash

echo "I am ${SLURM_PROCID} running on:"
hostname

#we are going to share an instance for Spark workers and HDFS datanodes
singularity instance start --bind $HOME/sparkhdfs/hadoop/logs:/opt/hadoop/logs,$HOME/sparkhdfs/hadoop/etc/hadoop:/opt/hadoop/etc/hadoop,$HOME/sparkhdfs/spark/conf:/opt/spark/conf,$HOME/sparkhdfs/spark/logs:/opt/spark/logs,$HOME/sparkhdfs/spark/work:/opt/spark/work \
 sparkhdfs.sif shinst

singularity run --bind $HOME/sparkhdfs/hadoop/logs:/opt/hadoop/logs,$HOME/sparkhdfs/hadoop/etc/hadoop:/opt/hadoop/etc/hadoop instance://shinst \
  sparkHDFSDatanode 2>&1 &

singularity run --bind $HOME/sparkhdfs/spark/conf:/opt/spark/conf,$HOME/sparkhdfs/spark/logs:/opt/spark/logs,$HOME/sparkhdfs/spark/work:/opt/spark/work instance://shinst \
  sparkWorker $SPARKMASTER -c 8 -m 12G


#the following without instance only Spark worker
#singularity run --bind $HOME/sparkhdfs/spark/conf:/opt/spark/conf,$HOME/sparkhdfs/spark/logs:/opt/spark/logs,$HOME/sparkhdfs/spark/work:/opt/spark/work sparkhdfs.sif \
# sparkWorker $SPARKMASTER -c 8 -m 8G 

EOF
chmod +x ${SPARK_LAUNCHER}

srun --exclusive -N 2 -n 2 -c 16 --ntasks-per-node=1 -l -o $HOME/sparkhdfs/SparkWorkers-`hostname`.out \
 ${SPARK_LAUNCHER} &

pid=$!
sleep 3600s
wait $pid

echo $HOME/sparkhdfs

echo "Ready Stopping SparkHDFS instances"

