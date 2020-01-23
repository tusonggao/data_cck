from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import numpy as np
import pandas as pd

df = pd.read_csv('./atad/train.csv')
print('df.shape is ', df.shape)

print('df.head(10) is ', df.head(10))

print('df.target 1 ratio is ', df.target.sum()/len(df))

