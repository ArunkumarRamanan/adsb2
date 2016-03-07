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

mkdir -p ws_cli ws_one ws_full0 ws_full1 ws_full2

function safe_run {
    DIR=$1
    WS=$2
    OP=$3
    XTOR=$4
    ROUND=$5
    for R in -1 $ROUND
    do
        ./touchup $DIR $WS $OP --xtor $XTOR --cohort --shuffle < TRAIN 2>&1 | tee $WS/$OP.log
    done


}

./touchup sum_study ws_cli train1 --xtor cli --cohort  --shuffle < TRAIN 2>&1 | tee ws_cli/train1.log
./touchup sum_study ws_cli train1 --xtor cli --cohort  --shuffle < TRAIN 2>&1 | tee ws_cli/train1.log
./touchup sum_study ws_cli train2 --xtor cli --cohort  --shuffle < TRAIN 2>&1 | tee ws_cli/train2.log

./touchup sum_study ws_cli pred --xtor cli --cohort  < TEST > ws_cli/pred 2> ws_cli/pred.log
./touchup sum_study ws_cli submit --xtor cli --cohort  < TEST > ws_cli/submit 2> ws_cli/submit.log

./touchup sum_study ws_one train1 --xtor one --cohort  --shuffle < TRAIN 2>&1 | tee ws_one/train1.log
./touchup sum_study ws_one train2 --xtor one --cohort  --shuffle < TRAIN 2>&1 | tee ws_one/train2.log
./touchup sum_study ws_one pred --xtor one --cohort  --fallback ws_cli/pred < TEST > ws_one/pred 2> ws_one/pred.log
./touchup sum_study ws_one submit --xtor one --cohort  --fallback ws_cli/pred < TEST > ws_one/submit 2> ws_one/submit.log

./touchup sum_study ws_full0 train1 --xtor full --cohort --shuffle < TRAIN 2>&1 | tee ws_full/train1.log
./touchup sum_study ws_full0 train2 --xtor full --cohort --shuffle < TRAIN 2>&1 | tee ws_full/train2.log
./touchup sum_study ws_full0 pred --xtor full --cohort --fallback ws_one/pred < TEST > ws_full/pred 2> ws_full/pred.log
./touchup sum_study ws_full0 submit --xtor full --cohort --fallback ws_one/pred < TEST > ws_full/submit 2> ws_full/submit.log


./touchup sum_study ws_full1 train1 --buddy dia --xtor full --cohort  --shuffle < TRAIN 2>&1 | tee ws_full1/train1.log
./touchup sum_study ws_full1 train2 --buddy dia --xtor full --cohort  --shuffle < TRAIN 2>&1 | tee ws_full1/train2.log

./touchup sum_study ws_full1 pred --buddy dia --xtor full --cohort  --fallback ws_one/pred < TEST > ws_full1/pred 2> ws_full1/pred.log
./touchup sum_study ws_full1 submit --buddy dia --xtor full --cohort  --fallback ws_one/pred < TEST > ws_full1/submit 2> ws_full1/submit.log

./touchup sum_study ws_full1 pred --buddy dia --xtor full --cohort  --fallback ws_one/pred --fallback2 ws_cli/pred < TEST > ws2_full/pred 2> ws2_full/pred.log
./touchup sum_study ws_full1 submit --buddy dia --xtor full --cohort  --fallback ws_one/pred --fallback2 ws_cli/pred < TEST > ws2_full/submit 2> ws2_full/submit.log


