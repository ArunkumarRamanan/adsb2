#!/bin/bash

BASE=$PWD

for P in sys dia
do
    cat TRAIN TEST | while read ID
    do
    mkdir -p $P/$ID
    qsub <<FOO
#PBS -l nodes=1:ppn=4
#PBS -j oe
#PBS -o $P/$ID.log
#PBS -S /bin/bash

cd $BASE
./study --preset $P raw/$ID/study $P/$ID 
FOO
    done
done

ln -s sys sum_study
