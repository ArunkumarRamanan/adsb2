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

./touchup sum_study pre/ws_cli pred --xtor cli --cohort  < TEST > pre/ws_cli/pred 2> pre/ws_cli/pred.log
./touchup sum_study pre/ws_cli submit --xtor cli --cohort  < TEST > pre/ws_cli/submit 2> pre/ws_cli/submit.log

./touchup sum_study pre/ws_one pred --xtor one --cohort  --fallback pre/ws_cli/pred < TEST > pre/ws_one/pred 2> pre/ws_one/pred.log
./touchup sum_study pre/ws_one submit --xtor one --cohort  --fallback pre/ws_cli/pred < TEST > pre/ws_one/submit 2> pre/ws_one/submit.log

./touchup sum_study pre/ws_full0 pred --xtor full --cohort --fallback pre/ws_one/pred < TEST > pre/ws_full0/pred 2> pre/ws_full0/pred.log
./touchup sum_study pre/ws_full0 submit --xtor full --cohort --fallback pre/ws_one/pred < TEST > pre/ws_full0/submit 2> pre/ws_full0/submit.log

./touchup sum_study pre/ws_full1 pred --buddy dia --xtor full --cohort  --fallback pre/ws_one/pred < TEST > pre/ws_full1/pred 2> pre/ws_full1/pred.log
./touchup sum_study pre/ws_full1 submit --buddy dia --xtor full --cohort  --fallback pre/ws_one/pred < TEST > pre/ws_full1/submit 2> pre/ws_full1/submit.log


