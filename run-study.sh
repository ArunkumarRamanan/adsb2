#!/bin/bash

mkdir sum_study
cat TRAIN TEST | while read a
do
    ./study raw/$a/study sum_study/$a
done
