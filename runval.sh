#!/bin/bash

rm -rf val
mkdir val
cat data/test_pid.txt | while read i
do
    ./study train/$i/study val/$i --decap -1
done
