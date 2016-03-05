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

rm -rf val1 val2 val3 val4

for i in 1 2 3 4
do
    mkdir val$i 2> /dev/null
    seq 1 500 | ./touchup sum_study val$i train1 $* --buddy dia --cohort --patch-cohort --shuffle --train TRAIN$i >& val$i/train1.log
    seq 1 500 | ./touchup sum_study val$i train2 $* --buddy dia --cohort --patch-cohort --shuffle --train TRAIN$i >& val$i/train2.log
    #seq 1 500 | ./touchup sum_study val$i submit $* --buddy dia --cohort --patch-cohort --train TRAIN$i 2> val$i/sub.err > val$i/sub
    seq 1 500 | ./touchup sum_study val$i eval $* --buddy dia --cohort --patch-cohort --train TRAIN$i 2> val$i/val.err | tee val$i/eval.log | tail -n 4 | head -n 1
done
