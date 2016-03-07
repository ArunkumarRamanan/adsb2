#!/bin/bash

if [ ! -f train.csv ]
then
    echo Cannot find TRAIN
    exit
fi

if [ ! -f TRAIN ]
then
    echo Cannot find TRAIN
    exit
fi

if [ ! -f TEST ]
then
    echo Cannot find TEST
    exit
fi

if [ ! -s raw -o ! -d raw ]
then
    echo Cannot find raw
    exit
fi

### Model-related parameters
### The same study directory should be run three times with different settings:
### 1.  ./study   input output  (optimized for overall accuracy)
### 2.  ./study --preset sys  input output  (optimized for systolic accuracy)
### 3.  ./study --preset dia  input output  (optimized for diastolic accuracy)

### Because the common computer vision part of the processing is very slow,
### we support writing the intermediate results to a snapshot file, which is
### used to speedup the computation of following two configurations.
### So the process becomes
### 1.  ./study   input output  --os snapshotfile
### 2.  ./study --preset sys  -s input_snapshot output
### 3.  ./study --preset dia  -s output_snapshot output
### The order of parameters does not matter.

### The output data of three configurations are stored as "sum_study", "sys" and "dia".
###

mkdir sum_study snapshot
cat TRAIN TEST | while read a
do
    ./study raw/$a/study sum_study/$a --os snapshot/$a
done

for P in sys dia
do
    mkdir $P
    cat TRAIN TEST | while read a
    do
        ./study -s --preset $P snapshot/$a $P/$a
        if [ ! -f $P/$a/report.txt ]
        then
            ./study --preset $P raw/$a/study $P/$a
        fi
    done
done

