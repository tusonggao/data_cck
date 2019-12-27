import os
import sys
import numpy as np
import pandas as pd

sys.setrecursionlimit(10000) 

gift_card = {0:0, 1:50, 2:50, 3:100, 4:200, 5:200, 6:300, 7:300, 8:400, 9:500, -1:500}
extra_cost_for_each_member = {0:0, 1:0, 2:9, 3:9, 4:9, 5:18, 6:18, 7:36, 8:36, 9:36+199, -1:36+398}

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
        days = np.random.randint(low=1, high=101, size=5000)
        assignment = {i:days[i] for i in range(0, 5000)}
        days_people_num, check = compute_days_people_num(assignment)
        print('check is ', check)
        if check is True:
            break
    score = compute_score(assignment, days_people_num)
    print('in generate_random_assignment, try_num: {} score: {:.2f} '.format(try_num, score))
    return assignment, score

def generate_BF_assignment():
    assignment_min, score_min = {}, float('inf')
    true_total_cnt, false_total_cnt = 0, 0
    assignment = {}
    def generate_BF_assignment_inner(family_id):
        nonlocal true_total_cnt, false_total_cnt
        if true_total_cnt >= 10:
            return
        for i in range(10):
            assignment[family_id] = choices[family_id][i]
            if family_id < 4999:
                generate_BF_assignment_inner(family_id+1)
            else:
                days_people_num, check = compute_days_people_num(assignment)
                if check is False:
                    false_total_cnt += 1
                    if false_total_cnt%1000==0:
                        print('false_total_cnt is ', false_total_cnt, 'true_total_cnt is ', true_total_cnt)
                else:
                    true_total_cnt += 1
                    score = compute_score(assignment, days_people_num)
                    print('get one score is', score)
                    if score < score_min:
                        score_min = score
                        assignment_min = assignment

    generate_BF_assignment_inner(0)

    print('in generate_BF_assignment, score_min: {:.2f} '.format(score_min))
    return assignment_min, score_min

#assignment, score = generate_random_assignment()
assignment, score = generate_BF_assignment()
outcome_df = pd.DataFrame({'family_id': list(assignment.keys()), 
                           'assigned_day': list(assignment.values())
             })
outcome_df = outcome_df[['family_id', 'assigned_day']]
outcome_df.to_csv('./submission/submission_tsg_{:.2f}.csv'.format(score), index=False)

print('prog ends here!')


#data = pd.read_csv('./atad/family_data.csv', index_col='family_id')
#data = pd.read_csv('./atad/family_data.csv')
#data = data.sort_values(by='n_people', ascending=False)
#data.to_csv('./atad/family_data_sorted.csv', index=False)
#print('total number of people is ', data['n_people'].sum())


#family_n_people = {}
#for family_id, n_people in zip(data['family_id'], data['n_people']):
#    family_n_people[family_id] = n_people
#file_w_content = 'family_id,assigned_day\n'

#file_w = open('./atad/submission_first_options_tsg.csv', 'w')
#file_w.write(file_w_content)
#file_w.close()
