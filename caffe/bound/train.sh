#!/usr/bin/env sh

export GLOG_log_dir=log
export GLOG_logtostderr=1

C=`basename $PWD`

if [ "$C" != adsb2 ]
then
    echo must run from adsb2
    exit
fi

cd caffe/bound

CAFFE=caffe

mkdir -p snapshots
#rm -rf snapshots/*

$CAFFE train --solver solver.prototxt $* 2>&1 | tee log

