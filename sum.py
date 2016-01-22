#!/usr/bin/env python

import sys
import pandas as pd
from tabulate import tabulate

x = pd.read_csv(sys.stdin, sep='\t', header=None)
print tabulate(x.describe())

