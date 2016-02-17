#!/bin/bash

C=`basename $PWD`

if [ "$C" != adsb2 ]
then
    echo must run from adsb2
    exit
fi

rm -rf caffe/bound_train/db
export OMP_NUM_THREADS=1
./import_many --list data/ctrs --train-list caffe/bound_train/train.list --output caffe/bound_train/db --replica 10
