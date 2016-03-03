#!/bin/bash

BASE=$PWD

cat TRAIN TEST | while read ID
do

mkdir -p sum_study/$ID
qsub <<FOO

#PBS -l nodes=1:ppn=4
#PBS -j oe
#PBS -o sum_study/$ID.log
#PBS -S /bin/bash

cd $BASE
./study raw/$ID/study sum_study/$ID --no-gif
FOO

done

