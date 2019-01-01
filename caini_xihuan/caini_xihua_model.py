import pandas as pd
import numpy as np

data_path = 'C:/D_Disk/data_competition/caini_xihuan/data/RSdata/'
df_test = pd.read_csv(data_path + 'test.csv')
df_train = pd.read_csv(data_path + 'train.csv')

print('df_train.shape is ', df_train.shape)
print('df_test.shape is ', df_test.shape)