#!/bin/bash

grep ", loss = " log | cut -f 7,10 -d ' '  | sed 's/,//'  > loss

gnuplot <<FOO
set terminal png
set output "loss.png"
set style data lp
#set yrange [0:1]
plot 'loss' using 1:2
FOO
cp loss.png ~/public_html/loss-contour.png
