#!/bin/bash

VALUE=$1
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TMPDIR="/tmp/ps2_${USER}/"

mkdir -p $TMPDIR
cd $TMPDIR

# Download the pre-trained model file
flock $TMPDIR -c "test ! -e resnet50_coco_best_v2.0.1.h5 && wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5"

OUTPUT_DIR=$SCRATCH/PS2/object_recognition_$SLURM_JOBID
mkdir -p $OUTPUT_DIR

source $SCRATCH/PS2/venv/bin/activate

python $SCRIPT_DIR/FirstDetection.py $VALUE $OUTPUT_DIR

