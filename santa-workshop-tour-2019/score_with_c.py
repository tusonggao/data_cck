import time
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import ctypes
from numpy.ctypeslib import ndpointer

lib = ctypes.CDLL('./score.so')
score = lib.score
# Define the types of the output and arguments of this function.
score.restype = ctypes.c_float
score.argtypes = [ndpointer(ctypes.c_int)]

print('get 111')

#sys.exit(0)

#sub = pd.read_csv('./submission/submission_672254.0276683343.csv')
sub = pd.read_csv('./submission/submission_best_69880.40.csv')
pred = np.int32(sub.assigned_day.values)
print('get 222')

start_t = time.time()
for i in range(1000):
    score_val = score(pred)
print('1000 times computation cost time: ', time.time() - start_t)

print('score of submission_672254.0276683343.csv is ', score_val)
print('get 333')
