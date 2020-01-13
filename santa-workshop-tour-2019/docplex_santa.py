import os
import sys
import numpy as np
import pandas as pd

def example_cplex(desired, n_people, has_accounting=True): # can't run on kaggle notebooks 
    def accounting_penalty(day, next_day):
        return (day-125)*(day**(0.5 + abs(day-next_day)/50.0))/400

    from docplex.mp.model import Model
    FAMILY_COST = np.asarray([0,50,50,100,200,200,300,300,400,500])
    MEMBER_COST = np.asarray([0, 0, 9,  9,  9, 18, 18, 36, 36,235])
    num_days = desired.max()
    num_families = desired.shape[0]
    solver = Model(name='Santa2019')
    solver.parameters.mip.tolerances.mipgap = 0.00
    solver.parameters.mip.tolerances.absmipgap = 0.00
    C = {}
    for fid, choices in enumerate(desired):
        for cid in range(10):
            C[fid, choices[cid]-1] = FAMILY_COST[cid] + n_people[fid] * MEMBER_COST[cid]

    B = solver.binary_var_dict(C, name='B')
    I = solver.integer_var_list(num_days, lb=125, ub=300, name='I')

    for day in range(num_days):
        solver.add(solver.sum(n_people[fid]*B[fid, day] for fid in range(num_families) if (fid,day) in B) == I[day])

    for fid in range(num_families):
        solver.add(solver.sum(B[fid, day] for day in range(num_days) if (fid,day) in B) == 1)

    preference_cost = solver.sum(C[fid, day]*B[fid, day] for fid, day in B)
    if has_accounting:
        Y = solver.binary_var_cube(num_days, 176, 176, name='Y')

        for day in range(num_days):
            next_day = np.clip(day+1, 0, num_days-1)
            gen = [(u,v) for v in range(176) for u in range(176)]
            solver.add(solver.sum(Y[day,u,v]*u for u,v in gen) == I[day]-125)
            solver.add(solver.sum(Y[day,u,v]*v for u,v in gen) == I[next_day]-125)
            solver.add(solver.sum(Y[day,u,v]   for u,v in gen) == 1)
            
        gen = [(day,u,v) for day in range(num_days) for v in range(176) for u in range(176)]
        accounting_penalties = solver.sum(accounting_penalty(u+125,v+125) * Y[day,u,v] for day,u,v in gen)
        solver.minimize(accounting_penalties+preference_cost)
    else:
        solver.minimize(preference_cost)

    solver.print_information()
    sol = solver.solve(log_output=True)
    if sol:
        print(sol.objective_value)
        assigned_days = np.zeros(num_families, int)
        for fid, day in C:
            if sol[B[fid, day]] > 0:
                assigned_days[fid] = day + 1
        return assigned_days

if __name__=='__main__':
    example_cplex(desired, n_people, has_accounting=True)


