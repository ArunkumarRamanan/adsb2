#!/bin/bash

C=`basename $PWD`

if [ "$C" != adsb2 ]
then
    echo must run from adsb2
    exit
fi

rm -rf caffe/contour/db
export OMP_NUM_THREADS=1
./import_many --list data/ctrs --polar --train-list data/train_pid.txt --output caffe/contour/db --replica 10
