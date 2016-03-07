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

function safe_train {
    DIR=$1
    WS=$2
    OP=$3
    XTOR=$4
    ROUND=$5
    shift 5
    ## we automatically tune # iterations of XGBoost
    ## but automatical tuning sometimes fails
    ## in such cases, a fixed fallback number of iteration is used to run XGBoost again
    for R in -1 $ROUND
    do
        mkdir -p $WS
        echo "./touchup --round1 $R --round2 $R $DIR $WS $OP --xtor $XTOR --cohort --shuffle $* < TRAIN 2>&1 | tee $WS/$OP.log" >> $WS/cmd
        ./touchup --round1 $R --round2 $R $DIR $WS $OP --xtor $XTOR --cohort --shuffle $* < TRAIN 2>&1 | tee $WS/$OP.log
        if [ $OP = train1 ]
        then
            if [ -f $WS/target.sys.0 -a -f $WS/target.sys.1 -a -f $WS/target.dia.0 -a -f $WS/target.dia.1 ]; then
                break
            fi
        fi
        if [ $OP = train2 ]
        then
            if [ -f $WS/error.dia -a -f $WS/error.sys ]; then
                break;
            fi
        fi
        echo Training failed with round=-1, falling back to fixed round.
        mv $WS $WS.failed
    done
}

safe_train sum_study ws_cli train1 cli 1000 
safe_train sum_study ws_cli train2 cli 200
./touchup sum_study ws_cli pred --xtor cli --cohort  < TEST > ws_cli/pred 2> ws_cli/pred.log
./touchup sum_study ws_cli submit --xtor cli --cohort  < TEST > ws_cli/submit 2> ws_cli/submit.log

safe_train sum_study ws_one train1 one 1000
safe_train sum_study ws_one train2 one 200
./touchup sum_study ws_one pred --xtor one --cohort  --fallback ws_cli/pred < TEST > ws_one/pred 2> ws_one/pred.log
./touchup sum_study ws_one submit --xtor one --cohort  --fallback ws_cli/pred < TEST > ws_one/submit 2> ws_one/submit.log

safe_train sum_study ws_full0 train1 full 1000
safe_train sum_study ws_full0 train2 full 200
./touchup sum_study ws_full0 pred --xtor full --cohort --fallback ws_one/pred --fallback2 ws_cli/pred < TEST > ws_full0/pred 2> ws_full0/pred.log
./touchup sum_study ws_full0 submit --xtor full --cohort --fallback ws_one/pred --fallback2 ws_cli/pred < TEST > ws_full0/submit 2> ws_full0/submit.log

safe_train sys ws_full1 train1 full 1500 --buddy dia  --fallback ws_one/pred --fallback2 ws_cli/pred 
safe_train sys ws_full1 train2 full 500 --buddy dia  --fallback ws_one/pred --fallback2 ws_cli/pred 
./touchup sys ws_full1 pred --buddy dia --xtor full --cohort --fallback ws_one/pred --fallback2 ws_cli/pred < TEST > ws_full1/pred 2> ws_full1/pred.log
./touchup sys ws_full1 submit --buddy dia --xtor full --cohort --fallback ws_one/pred --fallback2 ws_cli/pred < TEST > ws_full1/submit 2> ws_full1/submit.log

