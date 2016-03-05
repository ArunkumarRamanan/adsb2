#!/bin/bash

for P in sys dia
do
mkdir $P
cat TRAIN TEST | while read a
do
    ./study --preset $P raw/$a/study $P/$a
done
done

ln -s sys sum_study
