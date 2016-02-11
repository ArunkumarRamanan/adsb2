#!/bin/bash

caffe/bound_gyf/import.sh
caffe/bound_gyf/train.sh
cp caffe/bound_gyf/snapshots/fcn_iter_50000.caffemodel models/bound/caffe.params
caffe/contour/import.sh
caffe/contour/train.sh
cp caffe/contour/snapshots/fcn_iter_50000.caffemodel models/contour/caffe.params



