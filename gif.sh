#!/bin/bash

ROOT=train

mkdir -p gif
rm gif/*
cut -f 1 f50/val.list | while read a
do
    DIR=`dirname $ROOT/$a`
    N=`echo $a |cut -f 1 -d /`
    echo $N $DIR
    echo ./scc $DIR --gif gif/$N.gif -D adsb2.dp.smooth=150  #> gif/$N.txt
    ./scc $DIR --gif gif/$N.gif -D adsb2.dp.smooth=200 -D adsb2.dp.wall=300 -D adsb2.dp.th=0.3  > gif/$N.txt
done

echo "<html><body><table>" > gif/index.html
ls gif | grep gif | while read a; do
echo "<tr><td>$a</td><td><img src='$a'/></td></tr>" >> gif/index.html
done
echo "</table></body></html>" >> gif/index.html
