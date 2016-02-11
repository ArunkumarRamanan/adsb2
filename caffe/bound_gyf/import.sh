#!/bin/bash

C=`basename $PWD`

if [ "$C" != adsb2 ]
then
    echo must run from adsb2
    exit
fi

rm -rf caffe/bound_gyf/db
./import --list data/bbox.txt --root train/ --train-list data/train_pid.txt --output caffe/bound_gyf/db --replica 10
