#!/bin/bash

C=`basename $PWD`

if [ "$C" != adsb2 ]
then
    echo must run from adsb2
    exit
fi


function import {
    ids=$1
    list=$2
    db=$3
    > $list
    cat $ids | while read a
    do
        find ../../image0/$a -type f | sed 's/$/\t0/' >> $list
        find ../../image1/$a -type f | sed 's/$/\t1/' >> $list
    done
    rm -rf $db
    echo convert_imageset --gray --resize_height=256 --resize_width=256 --shuffle ./ $list $db
    convert_imageset --gray --resize_height=256 --resize_width=256 --shuffle ./ $list $db
}

rm -rf image0
rm -rf image1
./dump-1245 data/train_pid.txt

cd caffe/top/

shuf ../../data/train_pid.txt > studies
head -n 20 studies > val
tail -n +21 studies > train

import train train.list train.db
import val val.list val.db

#./import --list data/contour.list --polar --root train/ --train-list data/train_pid.txt --output caffe/contour/db --replica 100
