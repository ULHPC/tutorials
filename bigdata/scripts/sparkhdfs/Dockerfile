# Start from a base image
FROM ubuntu:18.04 AS builder

# Avoid prompts with tzdata
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /usr/local

# Update Ubuntu Software repository
RUN apt-get update
RUN apt-get install -y curl unzip zip

# Install wget
RUN apt-get install -y wget

# Download Apache Hadoop
RUN wget https://dlcdn.apache.org/hadoop/core/hadoop-3.3.6/hadoop-3.3.6.tar.gz
RUN tar xvf hadoop-3.3.6.tar.gz 
RUN mv hadoop-3.3.6 hadoop

# Download Apache Spark
RUN wget https://dlcdn.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
RUN tar xvf spark-3.5.0-bin-hadoop3.tgz
RUN mv spark-3.5.0-bin-hadoop3 spark

# Final stage
FROM ubuntu:18.04

COPY --from=builder \
/usr/local/hadoop /opt/hadoop

# Set environment variables for Hadoop
ENV HADOOP_HOME=/opt/hadoop
ENV PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

COPY --from=builder \
/usr/local/spark /opt/spark

# Set environment variables for Spark
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

# Install ssh and JDK11
RUN apt-get update && apt-get install -y openssh-server ca-certificates-java openjdk-11-jdk
ENV JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"

RUN ssh-keygen -t rsa -f /root/.ssh/id_rsa -q -P ""
RUN cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys
RUN chmod 0600 /root/.ssh/authorized_keys
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config

# Copy the docker-entrypoint.sh script into the Docker image
COPY ./docker-entrypoint.sh /

# Set the entrypoint script to run when the container starts
ENTRYPOINT ["/docker-entrypoint.sh"]

# Expose the necessary ports
EXPOSE 50070 8080 7078 22 9000 8020
### DONE
