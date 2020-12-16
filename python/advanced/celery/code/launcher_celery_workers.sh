#!/bin/bash -l
#
#SBATCH -N1
#SBATCH -c28
#SBATCH -J celery_workers

module load lang/Python/3.6.0-foss-2017a
source venv/bin/activate
celery -A ulhpccelery worker
