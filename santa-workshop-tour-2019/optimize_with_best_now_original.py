import os
import sys
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

MAX_BEST_CHOICE = 8
NUM_SWAP = 2500

#MAX_BEST_CHOICE = 8
#NUM_SWAP = 250

NUM_SECONDS = 3600
NUM_THREADS = 6
NUMBER_FAMILIES = 5000
NUMBER_DAYS = 100

COST_PER_FAMILY = [0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500]
COST_PER_FAMILY_MEMBER = [0, 0, 9, 9, 9, 18, 18, 36, 36, 36+199, 36+398]

DESIRED = {}
N_PEOPLE = {}
with open('./atad/family_data.csv') as file_r:
    line_cnt = 0
    for line in file_r:
        line_cnt += 1
        if line_cnt==1:
            continue
        family_id = int(line.strip().split(',')[0])
        n_people = int(line.strip().split(',')[-1])
        N_PEOPLE[family_id] = n_people
        DESIRED[family_id] = [int(day) for day in line.strip().split(',')[1:-1]]

def get_daily_occupancy(assigned_days):
    print('in get_daily_occupancy')
    daily_occupancy = {}
    for day in assigned_days:
        daily_occupancy[day] = daily_occupancy.get(day, 0) + 1
    daily_occupancy = np.array([daily_occupancy[i+1] for i in range(NUMBER_DAYS)])
    return daily_occupancy

#df = pd.read_csv('./mission/submission_672254.0276683343.csv')
#df = pd.read_csv('./mission/submission_best_69880.40.csv')
df = pd.read_csv('./atad/sample_submission.csv')
assigned_days = df['assigned_day'].values
daily_occupancy = get_daily_occupancy(assigned_days)
print('get daily_occupancy is ', daily_occupancy)

for i in range(40):
    print('i is', i)
    solver = pywraplp.Solver('Optimization preference cost', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    daily_occupancy = get_daily_occupancy(assigned_days).astype(float)
    fids = np.random.choice(range(NUMBER_FAMILIES), NUM_SWAP, replace=False)
    PCOSTM, B = {}, {}
    for fid in range(NUMBER_FAMILIES):
        if fid in fids:
            for i in range(MAX_BEST_CHOICE):
                PCOSTM[fid, DESIRED[fid][i]-1] = COST_PER_FAMILY[i] + N_PEOPLE[fid] * COST_PER_FAMILY_MEMBER[i]
                B[     fid, DESIRED[fid][i]-1] = solver.BoolVar('')
        else:
            daily_occupancy[assigned_days[fid]-1] -= N_PEOPLE[fid]

    solver.set_time_limit(NUM_SECONDS*NUM_THREADS*1000)
    solver.SetNumThreads(NUM_THREADS)

    for day in range(NUMBER_DAYS):
        if daily_occupancy[day]:
            solver.Add(solver.Sum([N_PEOPLE[fid] * B[fid, day] for fid in range(NUMBER_FAMILIES) if (fid,day) in B]) == daily_occupancy[day])
        
    for fid in fids:
        solver.Add(solver.Sum(B[fid, day] for day in range(NUMBER_DAYS) if (fid, day) in B) == 1)

    solver.Minimize(solver.Sum(PCOSTM[fid, day] * B[fid, day] for fid, day in B))
    sol = solver.Solve()
    
    status = ['OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNBOUNDED', 'ABNORMAL', 'MODEL_INVALID', 'NOT_SOLVED']
    if status[sol] in ['OPTIMAL', 'FEASIBLE']:
        tmp = assigned_days.copy()
        for fid, day in B:
            if B[fid, day].solution_value() > 0.48:
                tmp[fid] = day+1
        if cost_function(tmp)[2] < cost_function(assigned_days)[2]:
            assigned_days = tmp
            submission['assigned_day'] = assigned_days
            submission.to_csv('submission.csv', index=False)
        print('Result:', status[sol], cost_function(tmp))
    else:
        print('Result:', status[sol])
