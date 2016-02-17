#!/bin/bash

grep ", loss = " log | cut -f 2 -d = > loss

gnuplot <<FOO
set terminal png
set output "loss.png"
set style data lp
set yrange [0:0.04]
plot 'loss'
FOO
cp loss.png ~/public_html
