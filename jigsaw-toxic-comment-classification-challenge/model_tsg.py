import numpy as np 
import pandas as pd

if __name__=='__main__':
    train_df = pd.read_csv('./atad/train.csv')
    print('train_df.shape is ', train_df.shape)
    print('train_df.head(10) is ', train_df.head(10))
