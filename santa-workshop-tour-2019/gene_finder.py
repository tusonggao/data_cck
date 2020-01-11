# https://www.kaggle.com/isaienkov/genetic-algorithm-basics
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

np.random.seed(666)

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv')
print('data is ', data)
 
matrix = data[['choice_0', 'choice_1', 'choice_2', 'choice_3', 'choice_4',
       'choice_5', 'choice_6', 'choice_7', 'choice_8', 'choice_9']].to_numpy()

best = pd.read_csv("../input/local1/sub1.csv")
best = best['assigned_day'].to_list()

chromosome = [0 for i in range(500000)]
for i in range(5000):
    chromosome[i*100+best[i]-1] = 1
    
population = []
population.append(chromosome)

# Any results you write to the current directory are saved as output.
