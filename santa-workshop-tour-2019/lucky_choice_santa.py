import os
import ctypes
from numpy.ctypeslib import ndpointer
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from numba import njit, prange

lib = ctypes.CDLL('./score_double.so')
cost_function = lib.score
#cost_function.restype = ctypes.c_float
cost_function.restype = ctypes.c_double
cost_function.argtypes = [ndpointer(ctypes.c_int)]

'''
score = []
sub = []
name = os.listdir('/kaggle/input/santa-public')
for item in name:
    score.append(int(item.split('_')[1].split('.')[0]))
    sub.append(pd.read_csv('../input/santa-public/'+item, index_col='family_id'))
print(np.min(score))
print(len(sub))
'''

# Set Choice Selection Range
top_k = 3

# Load Data
data = pd.read_csv('./atad/family_data.csv', index_col='family_id')
submission = pd.read_csv(f'./submission/submission_69818.70.csv', index_col='family_id')
#submission = pd.read_csv(f'./submission_newest.csv', index_col='family_id')

sub = [submission]

print('get here 111')

# Run it on default submission file
original = submission['assigned_day'].values
original_score = cost_function(np.int32(original))
print('original_score is ', original_score)

choice_matrix = data.loc[:, 'choice_0': 'choice_9'].values

fam_weight = []

for i, s in enumerate(submission.iterrows()):
    for c in range(choice_matrix.shape[1]):
        if s[1].values==choice_matrix[i, c]:
            fam_weight.append(c+1)
fam_weight = np.array(fam_weight)
fam_weight = fam_weight / sum(fam_weight)
print(fam_weight)


# The redundancy is used to ensure evey choice can be selected with some probability since some choices are not selected in history submission
redundancy = 5 # any number larger than 0
choice_weight = np.zeros((5000, top_k))

for i in tqdm(range(5000)):
    for j in range(top_k):
        for s in sub:
            if choice_matrix[i, j] == s.loc[i, 'assigned_day']:
                choice_weight[i, j] += 1
                
choice_weight += redundancy
for j in range(choice_weight.shape[0]):
    choice_weight[j] /= sum(choice_weight[j])
    
print(choice_weight)

print('get here 222')

# A fast function for sampling indices from a 2-D probability array in a vectorised way
def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)

def lucky_choice_search(top_k, fam_size, original, choice_matrix, 
                   disable_tqdm=False, n_iter=100000000, 
                   verbose=10000, random_state=2019):
    
    best = original.copy()
    best_score = cost_function(np.int32(best))
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Select fam_size families from 5000 families with probability distribution fam_weight
    fam_indices = np.random.choice(range(choice_matrix.shape[0]), size=fam_size, p=fam_weight)

    for i in tqdm(range(n_iter), disable=disable_tqdm):
        if i%500000==0:
            print('i is', i)
        new = best.copy()
        
        # Select choices for each family based on the probability distribution of their choices from multiple history submissions
        new[fam_indices] = choice_matrix[fam_indices, random_choice_prob_index(choice_weight[fam_indices])]
        new_score = cost_function(np.int32(new))

        if new_score < best_score:
            best_score = new_score
            best = new
            print(f'{i} NEW BEST SCORE: ', best_score)
            submission['assigned_day'] = best
            submission.to_csv(f'./submission/random_best/submission_{best_score}.csv')

        if verbose and i % verbose == 0:
            print(f"Iteration #{i}: Best score is {best_score:.2f}")
            
    return best, best_score

best, best_score = lucky_choice_search(
    choice_matrix=choice_matrix, 
    top_k=top_k,
    fam_size=20, 
    original=original, 
    n_iter=2500000000, # run more iterations and find the optimal if you are lucky enough ;)
    disable_tqdm=False,
    #random_state=20191217,
    random_state=None,
    verbose=None
)

submission['assigned_day'] = best
submission.to_csv(f'./submission/random_best/submission_{best_score}.csv')

