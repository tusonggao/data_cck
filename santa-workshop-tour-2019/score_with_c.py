import random
import time
import os
import sys
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import ctypes
from numpy.ctypeslib import ndpointer

def get_all_files(dir_name):   
    all_files_lst = []
    def get_all_files_worker(path):
        allfilelist = os.listdir(path)
        for file in allfilelist:
            filepath = os.path.join(path, file)
            if os.path.isdir(filepath):
                get_all_files_worker(filepath)
            else:
                all_files_lst.append(filepath)
    get_all_files_worker(dir_name)
    return all_files_lst


def add_up_to_python(num):
    sum_v = 0
    for i in range(num):
        sum_v += i
    return sum_v

lib = ctypes.CDLL('./score_double.so')
score = lib.score
add_up_to = lib.add_up_to
score.restype = ctypes.c_double
score.argtypes = [ndpointer(ctypes.c_int)]
add_up_to.restype = ctypes.c_longlong
add_up_to.argtypes = [ctypes.c_long]

#sub = pd.read_csv('./submission/submission_672254.0276683343.csv')
#sub = pd.read_csv('./submission/submission_69818.70.csv')
#sub = pd.read_csv('./atad/sample_submission.csv')
#sub = pd.read_csv('./mission/random_best/submission_69752.88.csv')
#pred = np.int32(sub.assigned_day.values)

for file_name in get_all_files('./mission/random_best/'): 
    sub = pd.read_csv(file_name)
    pred = np.int32(sub.assigned_day.values)
    score_val = score(pred)
    print('file_name score is ', file_name, 'score_val is ', score_val)

print('get 222')

sys.exit(0)

N_NUM = 1000
start_t = time.time()
for i in range(N_NUM):
    score_val = score(pred)
sec_per_run = (time.time() - start_t)/N_NUM
print('score_val is ', score_val)
print('sec_per_run is:', sec_per_run)


best_score = score(pred)
print('original best_score is ', best_score)

iter_num = 0
computed_pairs = set()

for i in range(1000):
    print('i is ', i)
    while True:
        idx1, idx2 = random.sample(range(5000), 2)
        if idx1==idx2:
            continue
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        if (idx1, idx2) not in computed_pairs:
            computed_pairs.add((idx1, idx2))
            break
        print('idx1 idx2: ', idx1, idx2)

    for j1, j2 in itertools.product(range(1, 101), range(1, 101)):
        iter_num += 1
        if iter_num%5000==0:
            print('iter_num is ', iter_num, 'idx1 idx2: ', idx1, idx2)
        prev_day_1, prev_day2 = pred[idx1], pred[idx2]
        pred[idx1], pred[idx2] = j1, j2

        new_score = score(pred)
        if new_score < best_score:
            print('new best score is ', new_score)
            best_score = new_score
            df = pd.DataFrame({'family_id': np.arange(5000), 'assigned_day': pred})
            df.to_csv('./submission/random_best/submission_{}.csv'.format(best_score), index=False)
        else:
            pred[idx1], pred[idx2] = prev_day_1, prev_day2


print('prog ends here!')

sys.exit(0)

            
pred_new = pred.copy()

fam = pd.read_csv("./atad/family_data.csv")
pref = fam.values[:,1:-1]
n_people = fam.n_people.values
fam_size_order = np.argsort(n_people)#[::-1]

best_score = score(pred)

for t in tqdm(range(20)):
    print(t,best_score,'     ',end='\r')
    for i in fam_size_order:
        for j in range(10):
            di = pred[i]
            pred[i] = pref[i,j]
            cur_score = score(pred)
            if cur_score < best_score:
                best_score = cur_score
            else:
                pred[i] = di

def opt():
    print('in opt()')
    best_score = score(pred)
    for i in tqdm(range(5000)):
        if (i%100==0):
            print(i,best_score,'     ',end='\r')
        for j in range(5000):
            di = pred[i]
            pred[i] = pred[j]
            pred[j] = di
            cur_score = score(pred)
            if cur_score < best_score:
                best_score = cur_score
            else: # revert
                pred[j] = pred[i]
                pred[i] = di
            
opt()
opt()

sub.assigned_day = pred
_score = score(pred)

sub.to_csv(f'submission_{_score}.csv',index=False)

sys.exit(0)

sub = pd.read_csv('./submission/submission_672254.0276683343.csv')
#sub = pd.read_csv('./atad/sample_submission.csv')
pred = np.int32(sub.assigned_day.values)
print('get 222')

sys.exit(0)


start_t = time.time()
for i in range(1):
    score_val = score(pred)
print('1000 times computation cost time: ', time.time() - start_t)

print('double score of submission_672254.0276683343.csv is ', score_val)
print('get 333')


start_t = time.time()
for i in range(1000):
    sum_v = add_up_to(100000)
c_cost_time = time.time() - start_t
print('sum_v of c is ', sum_v)

start_t = time.time()
for i in range(1000):
    sum_v = add_up_to_python(100000)
python_cost_time = time.time() - start_t
print('sum_v of python is ', sum_v)

start_t = time.time()
for i in range(1000):
    sum_v = np.arange(100000).sum()
np_cost_time = time.time() - start_t
print('sum_v of np is ', sum_v)

print('python add up to', sum_v, 'cost time: ', 'c_cost_time: {} python_cost_time: {} np_cost_time: {}'.format(c_cost_time, python_cost_time, np_cost_time))


lib = ctypes.CDLL('./score_float.so')
score = lib.score
# Define the types of the output and arguments of this function.
score.restype = ctypes.c_float
score.argtypes = [ndpointer(ctypes.c_int)]

