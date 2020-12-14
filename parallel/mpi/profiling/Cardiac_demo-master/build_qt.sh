#!/bin/bash
#Copyright (c) 2016, Intel Corporation

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

QT_HOME=/usr
QMAKE=$QT_HOME/bin/qmake-qt4

cd $DIR
if [ ! -d $DIR/build ]; then
    mkdir build
fi
cd $DIR/build

export HEART_EXTRA="-DENABLE_GL=1 -L$DIR/build"

$QMAKE ../heart_demo.pro
make clean

icc ../rcm.cpp -I.. -shared -fPIC -o librcm.so

if [ $? != 0 ]; then
    echo "Error building librcm.so"
    exit 1
fi

make -j16

