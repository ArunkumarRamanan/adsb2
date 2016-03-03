How to produce testing results.

1. Data Preparation

Training and validation studies are numbered from 1-700.
It is assumed that testing data are also similarly numbered, not using
numbers between 1-700 which are already used.

Prepare a file named TEST and list the test study numbers, like the
provided TRAIN FILE.

Prepare a directory (or a symbolic link to a directory) named "raw",
containing all the training, validating and testing data, like the
following structure:

raw/1/study/2ch_21/...
raw/1/study/sax_10/...
...
raw/700/study/2ch_15/..
raw/700/study/sax_10/..
....

2. Process all the study files.

Run command like below for each study

./study raw/1/study sum_study/1
./study raw/2/study sum_study/2
...

This can be done using the provided run-study.sh or qsub-study.sh to
parallel process using PBS.  This is the time comsuming step.

3. Cross validation

./run-val.sh

4. Producing submission file.

./run-submit.sh

The produced file for submission is in ws_full/submit



