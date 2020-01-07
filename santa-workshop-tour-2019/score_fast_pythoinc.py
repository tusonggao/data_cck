import time
import os
import sys
import numpy as np
import pandas as pd
from numba import njit, prange

data = pd.read_csv('./atad/family_data.csv', index_col='family_id')
prediction = pd.read_csv("./submission/submission_69818.70.csv", index_col='family_id').assigned_day.values
family_size = data.n_people.values.astype(np.int8)

print('get here 111')

penalties = np.asarray([
    [
        0,
        50,
        50 + 9 * n,
        100 + 9 * n,
        200 + 9 * n,
        200 + 18 * n,
        300 + 18 * n,
        300 + 36 * n,
        400 + 36 * n,
        500 + 36 * n + 199 * n,
        500 + 36 * n + 398 * n
    ] for n in range(family_size.max() + 1)
])
family_cost_matrix = np.concatenate(data.n_people.apply(lambda n: np.repeat(penalties[n, 10], 100).reshape(1, 100)))
for fam in data.index:
    for choice_order, day in enumerate(data.loc[fam].drop("n_people")):
        family_cost_matrix[fam, day - 1] = penalties[data.loc[fam, "n_people"], choice_order]


accounting_cost_matrix = np.zeros((500, 500))
for n in range(accounting_cost_matrix.shape[0]):
    for diff in range(accounting_cost_matrix.shape[1]):
        accounting_cost_matrix[n, diff] = max(0, (n - 125.0) / 400.0 * n**(0.5 + diff / 50.0))

print('get here 222')

sys.exit(0)


@njit(fastmath=True)
def cost_function(prediction, family_size, family_cost_matrix, accounting_cost_matrix):
    N_DAYS = family_cost_matrix.shape[1]
    MAX_OCCUPANCY = 300
    MIN_OCCUPANCY = 125
    penalty = 0
    daily_occupancy = np.zeros(N_DAYS + 1, dtype=np.int16)
    for i, (pred, n) in enumerate(zip(prediction, family_size)):
        daily_occupancy[pred - 1] += n
        penalty += family_cost_matrix[i, pred - 1]

    accounting_cost = 0
    n_low = 0
    n_high = 0
    daily_occupancy[-1] = daily_occupancy[-2]
    for day in range(N_DAYS):
        n_next = daily_occupancy[day + 1]
        n = daily_occupancy[day]
        n_high += (n > MAX_OCCUPANCY) 
        n_low += (n < MIN_OCCUPANCY)
        diff = abs(n - n_next)
        accounting_cost += accounting_cost_matrix[n, diff]

    return np.asarray([penalty, accounting_cost, n_low, n_high])

print('get here 333')

family_size = family_size.astype(np.int16)

start_t = time.time()
for i in range(1):
    score = cost_function(prediction, family_size, family_cost_matrix, accounting_cost_matrix)
print('score is ', score, 'cost time: ', time.time() - start_t)

print('get here 444')






