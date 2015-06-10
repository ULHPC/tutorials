#!/bin/bash -l

# Root work directory
ROOT=$WORK/PS2

# Temporary directory in /tmp
TEMP=`mktemp -d`

CONFIG_TARBALL=$ROOT/jcell/config.tgz
RESULT_DIR=$ROOT/jcell/results

cd $TEMP

tar xzvf $CONFIG_TARBALL -C $TEMP $1

mkdir -p $RESULT_DIR

# If the result file exist, exit (and do not compute the file again)
if [[ -f $RESULT_DIR/$1.result ]] ; then
  exit 0
fi

cd $ROOT/jcell/JCell/bin
# Set the classpath (from an exisiting script
eval `sed 's/\/tmp\/bernabe/$PWD/g' setClasspathCronos`
# Launch JCell
java JCell $TEMP/$1 2>&1 > $TEMP/$1.result

# If the execution is successful,
if [[ $? -eq 0 ]] ; then
  mv $TEMP/$1.result $RESULT_DIR/$1.result
fi

