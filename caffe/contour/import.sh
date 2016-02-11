#!/bin/bash

C=`basename $PWD`

if [ "$C" != adsb2 ]
then
    echo must run from adsb2
    exit
fi

rm -rf caffe/contour/db
./import --list data/contour.list --polar --root train/ --train-list data/train_pid.txt --output caffe/contour/db --replica 100
