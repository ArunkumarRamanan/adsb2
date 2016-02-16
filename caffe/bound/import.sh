#!/bin/bash

C=`basename $PWD`

if [ "$C" != adsb2 ]
then
    echo must run from adsb2
    exit
fi

rm -rf caffe/bound/db
export OMP_NUM_THREADS=1
./import_many --list data/ctrs --train-list data/train_pid.txt --output caffe/bound/db --replica 10
