import os
import time
import sys
import numpy as np
import pandas as pd
from collections import Counter

df = pd.read_csv('./atad/family_data.csv')
choice_matrix = df.loc[:, 'choice_0': 'choice_9'].values
print('choice_matrix.shape is', choice_matrix.shape)

sub = pd.read_csv('./mission/submission_69707.687.csv')
assigned_days = sub['assigned_day'].values
print('assigned_days.shape is ', assigned_days.shape)

idx_lst = []

for i in range(choice_matrix.shape[0]):
    if assigned_days[i] not in list(choice_matrix[i]):
        print('not in ')
    else:
        idx = list(choice_matrix[i]).index(assigned_days[i])
        idx_lst.append(idx)
        
print('prog ends here!')
print('idx_lst is ', idx_lst)
print('counter of idx_lst is ', Counter(idx_lst))
print('max of idx_lst is ', max(idx_lst))

