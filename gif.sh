#!/bin/bash

ROOT=/ssd/wdong/adsb2/train

mkdir gif
cut -f 1 f50/val.list | while read a
do
    DIR=`dirname $ROOT/$a`
    N=`echo $a |cut -f 1 -d /`
    echo $N $DIR
    ../detect $DIR --gif gif/$N.gif
done

echo "<html><body><table>" > gif/index.html
ls gif | grep gif | while read a; do
echo "<tr><td>$a</td><td><img src='$a'/></td></tr>" >> gif/index.html
done
echo "</table></body></html>" >> gif/index.html
