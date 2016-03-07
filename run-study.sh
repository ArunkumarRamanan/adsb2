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

