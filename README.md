#

## Check Configuration

Check the content of file adsb2.xml. The path specified
in data_dir should contain the training data.

## import images
# after that, f50/train and f50/val will be produced
# these dirs will be used by Caffe
```
./import --list bbox.txt -f 50 f50
```

## Generate Caffe Training Directory

```
caffex-fcn/finetune-init.py
```
This will produce a file named config.json.
Edit the file, change "val_source" to "f50/val"
and "train_source" to "f50/train".  These are the two
database directories produced in the import step.

## Train Caffe

```
caffex-fcn/fintune-generate.py
./train.sh	
```
This step will produce the following files in the current
directory.  If you have files of the same name, they'll be
overwritten:

- solver.prototxt
- train.prototxt
- val.prototxt
- train.sh
- model
- log
- snapshots

After traing is done, copy one of the caffemodel files
into model/caffe.params.  The model directory should then
be ready to go.

## Testing Model

```
./detect sax --gif x.gif
```

