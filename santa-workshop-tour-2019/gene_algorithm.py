import os
import time
import sys
import ctypes
import random
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from numpy.ctypeslib import ndpointer

sys.setrecursionlimit(10000) 

def get_all_files(dir_name):   # 递归得到文件夹下的所有文件
    all_files_lst = []
    def get_all_files_worker(path):
        allfilelist = os.listdir(path)
        for file in allfilelist:
            filepath = os.path.join(path, file)
            #判断是不是文件夹
            if os.path.isdir(filepath):
                get_all_files_worker(filepath)
            else:
                all_files_lst.append(filepath)
    get_all_files_worker(dir_name)
    return all_files_lst

start_t_global = time.time()

lib = ctypes.CDLL('./score_double.so')
score = lib.score
add_up_to = lib.add_up_to
score.restype = ctypes.c_double
score.argtypes = [ndpointer(ctypes.c_int)]
add_up_to.restype = ctypes.c_longlong
add_up_to.argtypes = [ctypes.c_long]

#current_min_score = 672254.027668334

class GA(object):
    def __init__(self, DNA_size, DNA_bound, cross_rate, mutation_rate, pop_size=10000):
        self.DNA_size = DNA_size
        DNA_bound[1] += 1
        self.DNA_bound = DNA_bound
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        self.population_set = set()
        #self.population = np.random.randint(*DNA_bound, size=(pop_size, DNA_size)).astype(np.int8)  # int8 for convert to ASCII

    def get_fitness(self):                      # count how many character matches
        fitness_lst = []
        for i in range(self.population_score_map
        
        
        match_count = (self.pop == TARGET_ASCII).sum(axis=1)
        return match_count

    def select(self):
        fitness = self.get_fitness() + 1e-4     # add a small amount to avoid all zero fitness
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness/fitness.sum())
        return idx

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)                        # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)   # choose crossover points
            parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
        return parent

    def mutate(self, child):
        mutate_choices = np.array([0, 1, 2, 3, 4])
        mutate_choices_weight = np.array([5, 4, 3, 2, 1])
        choice_idx = np.random.choice(mutate_choices, size=5000, replace=True, 
                         p=mutate_choices_weights/mutate_choices_weights.sum())
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                child[point] = choices[family_id, idx[family_id]]
        return child

    def evolve(self):
        pop = self.select()
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
        for 
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop


data = pd.read_csv('./atad/family_data.csv')
print('data.head(5) is ', data.head(5))
#choices = data.loc[:, [choice_0,choice_1,choice_2,choice_3,choice_4,choice_5,choice_6,choice_7,choice_8, 'choice_9']].values.astype(np.int32)
choices = data.loc[:,'choice_0':'choice_9'].values.astype(np.int32)
print('choices.shape is ', choices.shape)

sys.exit(0)

def store_assignment(assignment, score, current_min=False):
    print('in store_assignment(), score is ', score)
    outcome_df = pd.DataFrame({'family_id': range(5000), 'assigned_day': list(assignment)})
    outcome_df.to_csv('./submission/min_gene/submission_tsg_{:.5f}.csv'.format(score), index=False)
    
df_seed1 = pd.read_csv('./mission/min_iterative/submission_tsg_358314.23221.csv')
assigned_days = df_seed1.assigned_day.values.astype(np.int32)

score_val = score(assigned_days)
print('score val of submission_tsg_358314.23221.csv is ', score_val)

sys.exit(0)
#df_seed2 = pd.read_csv('./submission/min_iterative/submission_tsg_358360.58547.csv')

N_GENERATIONS = 2
for generation in range(N_GENERATIONS):
        fitness = ga.get_fitness()
        best_DNA = ga.pop[np.argmax(fitness)]
        best_phrase = ga.translateDNA(best_DNA)
        print('Generation: ', generation, ': ', best_phrase)
        if best_phrase == TARGET_PHRASE:
            break
        ga.evolve()

print('prog ends here! total cost time: ', time.time() - start_t_global)

#########################################################################################################

