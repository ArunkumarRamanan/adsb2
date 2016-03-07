# ADSB2: Annual Data Science Bow

## How to run model on test set.

Our model consists of a set of binary programs (study, touch) and
a set of offline pre-trained model files (models/).  These
pre-trained models are frozen at model submission and are not
to be retrained for the final test set.

### System Dependency

Hardware: any x86_64 machine with > 16GB memory.
Software: 64-bit Linux with a modern kernel (Centos > 2.6, Ubuntu > 12.04)

The package doesn't depend on other software to produce the submission
files.  The programs needs ImageMagick to produce visualization.

### Data Preparation

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

There is a train.csv file in the directory, which contains the
groundtruth data.  When validation set is released, the groundtruth
data should be merged into this file.

### Running the Programs

```
./run-study.sh
./run-submit.sh
```

The first script processes each study directory and produce
initial predictions.  The second script post-process the
initial predictions with linear regression.  The second
script is capable of handling cases when the first script
fails at certain models.

The second script produces a series of submission files:

ws_full2/submit
ws_full1/submit
ws_full0/submit
ws_one/submit
ws_cli/submit

The top 2 of the successfully produced submit files shall
be used for final submission. (If run-submit.sh fails
due to non-model-related issues, justifiable means
should be used to recover the failure before resorting to
an sub-optimal submission file).

## Visualizing Predictions

!!! This is not part of our model submission.

```
./study .../10/study  output --gif
```

Then use a browser to open output/index.html to see the
visualization.



