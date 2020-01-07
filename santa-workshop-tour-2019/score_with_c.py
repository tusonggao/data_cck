import time
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import ctypes
from numpy.ctypeslib import ndpointer

lib = ctypes.CDLL('./score_float.so')
score = lib.score
# Define the types of the output and arguments of this function.
score.restype = ctypes.c_float
score.argtypes = [ndpointer(ctypes.c_int)]

print('get 111')

#sys.exit(0)

sub = pd.read_csv('./submission/submission_672254.0276683343.csv')
#sub = pd.read_csv('./submission/submission_best_69880.40.csv')
#sub = pd.read_csv('./atad/sample_submission.csv')
pred = np.int32(sub.assigned_day.values)
print('get 222')

start_t = time.time()
for i in range(1):
    score_val = score(pred)
print('1000 times computation cost time: ', time.time() - start_t)

print('float score of submission_672254.0276683343.csv is ', score_val)
print('get 333')

lib = ctypes.CDLL('./score_double.so')
score = lib.score
# Define the types of the output and arguments of this function.
#score.restype = ctypes.c_float
score.restype = ctypes.c_double
score.argtypes = [ndpointer(ctypes.c_int)]

print('get 111')

#sys.exit(0)

sub = pd.read_csv('./submission/submission_672254.0276683343.csv')
#sub = pd.read_csv('./atad/sample_submission.csv')
pred = np.int32(sub.assigned_day.values)
print('get 222')

start_t = time.time()
for i in range(1):
    score_val = score(pred)
print('1000 times computation cost time: ', time.time() - start_t)

print('double score of submission_672254.0276683343.csv is ', score_val)
print('get 333')
