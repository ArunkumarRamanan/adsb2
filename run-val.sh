#!/bin/bash

if [ ! -s raw -o ! -d raw ]
then
    echo Cannot find raw
    exit
fi

if [ ! -d sum_study ]
then
    echo Cannot sum_study
    exit
fi

for i in 1 2 3
do
    mkdir val$i
    echo train1
    seq 1 500 | ./touchup sum_study val$i train1 $* --cohort --shuffle --train TRAIN$i >& val$i/train1.log
    echo train2
    seq 1 500 | ./touchup sum_study val$i train2 $* --cohort --shuffle --train TRAIN$i >& val$i/train2.log
    echo eval
    seq 1 500 | ./touchup sum_study val$i eval $* --cohort | tee val$i/eval.log | tail -n 4
done
