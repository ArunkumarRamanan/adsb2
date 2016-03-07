How to produce testing results.

1. Data Preparation


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


Source Code

git@bitbucket.org:wdong397/adsb2.git


