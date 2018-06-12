#!/bin/bash -l
#
#SBATCH -N1
#SBATCH -c1
#SBATCH -J redis-server

# Generate a port number between 64000 and 64999
# with last 3 digits corresponding to $SLURM_JOB_ID
PORT=$(($SLURM_JOB_ID % 1000 + 64000))
# IP of the node
REDIS_SERVER_IP=$(facter ipaddress)
# Retrieve password in redis config file
PASSWORD=$(grep '^requirepass' ./redis.conf | sed 's/requirepass \(.*\)/\1/g')

# Write the Redis server parameters
# so that celery workers know about it
cat > celery.ini << EOF
[redis]
broker_password=$PASSWORD
broker_hostname=$REDIS_SERVER_IP
broker_port=$PORT
EOF

# Run redis server on the port
./redis*/src/redis-server ./redis.conf --port $PORT --bind $REDIS_SERVER_IP
