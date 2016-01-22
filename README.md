# ADSB2: Annual Data Science Bow

## Build programs

```
git submodule init
git submodule update
ln -s Makefile.shared Makefile
make 
```

## import images

The bbox.txt is a bounding box listing file.

```
./import --root /ssd/wdong/adsb2/train/ -f 50 bbox.txt f50
```

After that, f50/train and f50/val will be produced
these dirs will be used by Caffe.
"-f" specify the number of fold.  The command
only generate one of the 50 cross validation
configurations.  To generate all, add "--full" to
command line.

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

To change Caffe setting, either edit config.json and run
finetune-generate.py again, or directly modify the Caffe
prototxt files (those will be overwritten by subsequent
runs of finetune-generate.py).

## Testing Model

```
./detect /ssd/wdong/adsb2/train/337/study/sax_17 --gif x.gif
```
The detect command assumes a "model" directory under the
current directory.  Otherwise, specify the model with
"-D adsb2.caffe.model=model-dir".

