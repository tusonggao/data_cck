import os
import time
import sys
import random
import numpy as np
import pandas as pd
sys.setrecursionlimit(10000) 

start_t_global = time.time()

current_min_score = 672254.027668334

gift_card = {0:0, 1:50, 2:50, 3:100, 4:200, 5:200, 6:300, 7:300, 8:400, 9:500, -1:500}
extra_cost_for_each_member = {0:0, 1:0, 2:9, 3:9, 4:9, 5:18, 6:18, 7:36, 8:36, 9:36+199, -1:36+398}

data = pd.read_csv('./atad/family_data.csv')
data = data.sort_values(by='n_people', ascending=False)
#family_ids_sorted = list(data['family_id'].values)
family_ids_sorted = list(range(5000))

choices, choices_reversed, family_people_num = {}, {}, {}
with open('./atad/family_data.csv') as file_r:
    line_cnt = 0
    for line in file_r:
        line_cnt += 1
        if line_cnt==1:
            continue
        family_id = int(line.strip().split(',')[0])
        n_people = int(line.strip().split(',')[-1])
        family_people_num[family_id] = n_people
        choices[family_id] = [int(day) for day in line.strip().split(',')[1:-1]]
        choices_reversed[family_id] = {}
        for n, day in enumerate(line.strip().split(',')[1:-1]):
            choices_reversed[family_id][int(day)] = n

def compute_days_people_num(assignment):
    days_people_num = {i:0 for i in range(1, 101, 1)}
    for family_id, day in assignment.items():
        days_people_num[day] += family_people_num[family_id] 
    check = all(125 <= days_people_num[i] <= 300 for i in range(1, 101, 1))
    return days_people_num, check

def compute_score(assignment, days_people_num):
    global gift_card, extra_for_each_member
    total_cost = 0
    for family_id, day in assignment.items():
        option_num = choices_reversed[family_id].get(day, -1)
        n_people = family_people_num[family_id]
        total_cost += gift_card[option_num] + extra_cost_for_each_member[option_num]*n_people
    for i in range(1, 101, 1):
        N_d = days_people_num[i]
        N_d_plus_1 = days_people_num[i+1] if i < 100 else days_people_num[i]
        total_cost += max(0, (N_d - 125) / 400 * (N_d ** (0.5 + abs(N_d - N_d_plus_1) / 50)))
    return total_cost

def generate_random_assignment():
    try_num = 0
    while True:
        try_num += 1
        #days = np.random.randint(low=1, high=101, size=5000)
        #assignment = {i:days[i] for i in range(0, 5000)}
        options = np.random.randint(low=0, high=10, size=5000)
        assignment = {i:choices[i][options[i]] for i in range(0, 5000)}
        days_people_num, check = compute_days_people_num(assignment)
        print('check is ', check)
        if check is True:
            break
    score = compute_score(assignment, days_people_num)
    print('in generate_random_assignment, try_num: {} score: {:.2f} '.format(try_num, score))
    return assignment, score

def store_assignment(assignment, score, current_min=False):
    print('in store_assignment(), score is ', score)
    outcome_df = pd.DataFrame({'family_id': list(assignment.keys()), 
                           'assigned_day': list(assignment.values())})
    outcome_df = outcome_df[['family_id', 'assigned_day']]
    outcome_df = outcome_df.sort_values(by='family_id')
    if current_min:
        outcome_df.to_csv('./submission/min/submission_tsg_{:.5f}.csv'.format(score), index=False)
    elif score < 700000:
        outcome_df.to_csv('./submission/genes/submission_tsg_{:.5f}.csv'.format(score), index=False)

def generate_fine_tuned_assignment():
    score_min = float('inf')
    assigned_days_lst = list(range(1, 101, 1))*50

    while True:
        random.shuffle(assigned_days_lst)
        assignment = {i:assigned_days_lst[i] for i in range(5000)}
        days_people_num, check = compute_days_people_num(assignment)
        if check is True:
            score = compute_score(assignment, days_people_num)
            score_min = score
            print('origin score_min is ', score_min)
            break

    family_ids = list(range(5000))
    random.shuffle(family_ids)
    for i in family_ids:
        for j in range(10):
            new_assignment = assignment.copy()
            new_assignment[i] = choices[i][j]
            days_people_num, check = compute_days_people_num(new_assignment)
            if check is True:
                score = compute_score(new_assignment, days_people_num)
                if score < 700000:   #672254.027668334
                    pass
                    #store_assignment(new_assignment, score)
                if score < score_min:
                    score_min = score
                    assignment = new_assignment.copy()
            else:
                pass
    print('final score_min is ', score_min)
    store_assignment(assignment, score_min, current_min=True)
    return assignment, score_min

df = pd.read_csv('./submission/submission_best.csv')
assignment = {family_id : day for family_id, day in 
              zip(df['family_id'].tolist(), df['assigned_day'].tolist())}
days_people_num, check = compute_days_people_num(assignment)
score = compute_score(assignment, days_people_num)
print('best score is ', score)
sys.exit(0)

#assignment, score = generate_random_assignment()
#assignment, score = generate_BF_assignment()
for try_num in range(1000):
    print('try_num is ', try_num)
    generate_fine_tuned_assignment()

print('prog ends here! total cost time: ', time.time() - start_t_global)

#########################################################################################################

#data.to_csv('./atad/family_data_sorted.csv', index=False)
#print('total number of people is ', data['n_people'].sum())

    #sample_submission = pd.read_csv('./atad/sample_submission.csv')
    #assignment = {family_id:day for family_id, day in 
    #              zip(sample_submission['family_id'].tolist(),
    #                  sample_submission['assigned_day'].tolist())}
