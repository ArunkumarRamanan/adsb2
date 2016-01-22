#!/bin/bash

ROOT=/ssd/wdong/adsb2/train

mkdir gif
cut -f 1 test.list | while read a
do
    DIR=`dirname $ROOT/$a`
    N=`echo $a |cut -f 1 -d /`
    echo $N $DIR
    ../detect -D adsb2.caffe.model=../model $DIR --gif gif/$N.gif
done

echo "<html><body><table>" > bbox/index.html
ls bbox | grep gif | while read a; do
echo "<tr><td>$a</td><td><img src='$a'/></td></tr>" >> bbox/index.html
done
echo "</table></body></html>" >> bbox/index.html
