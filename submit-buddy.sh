#!/bin/bash

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

if [ ! -d sum_study ]
then
    echo Cannot sum_study
    exit
fi
mkdir -p ws_cli ws_one ws_full

./touchup sum_study ws_cli train1 --xtor cli --cohort --patch-cohort --shuffle < TRAIN 2>&1 | tee ws_cli/train1.log
./touchup sum_study ws_cli train1 --xtor cli --cohort --patch-cohort --shuffle < TRAIN 2>&1 | tee ws_cli/train1.log
./touchup sum_study ws_cli train2 --xtor cli --cohort --patch-cohort --shuffle < TRAIN 2>&1 | tee ws_cli/train2.log
./touchup sum_study ws_cli pred --xtor cli --cohort --patch-cohort < TEST > ws_cli/pred 2> ws_cli/pred.log
./touchup sum_study ws_cli submit --xtor cli --cohort --patch-cohort < TEST > ws_cli/submit 2> ws_cli/submit.log

./touchup sum_study ws_one train1 --xtor one --cohort --patch-cohort --shuffle < TRAIN 2>&1 | tee ws_one/train1.log
./touchup sum_study ws_one train2 --xtor one --cohort --patch-cohort --shuffle < TRAIN 2>&1 | tee ws_one/train2.log
./touchup sum_study ws_one pred --xtor one --cohort --patch-cohort --fallback ws_cli/pred < TEST > ws_one/pred 2> ws_one/pred.log
./touchup sum_study ws_one submit --xtor one --cohort --patch-cohort --fallback ws_cli/pred < TEST > ws_one/submit 2> ws_one/submit.log

./touchup sum_study ws_full train1 --buddy dia --xtor full --cohort --patch-cohort --shuffle < TRAIN 2>&1 | tee ws_full/train1.log
./touchup sum_study ws_full train2 --buddy dia --xtor full --cohort --patch-cohort --shuffle < TRAIN 2>&1 | tee ws_full/train2.log
./touchup sum_study ws_full pred --buddy dia --xtor full --cohort --patch-cohort --fallback ws_one/pred < TEST > ws_full/pred 2> ws_full/pred.log
./touchup sum_study ws_full submit --buddy dia --xtor full --cohort --patch-cohort --fallback ws_one/pred < TEST > ws_full/submit 2> ws_full/submit.log

