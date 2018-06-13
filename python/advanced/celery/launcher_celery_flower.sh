#!/bin/bash -l
#
#SBATCH -N1
#SBATCH -c1
#SBATCH -J flower

module load lang/Python/3.6.0-foss-2017a
source venv/bin/activate
celery -A ulhpccelery flower --address="$(facter ipaddress)"
