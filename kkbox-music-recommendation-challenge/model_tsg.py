from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import numpy as np
import pandas as pd

train_df = pd.read_csv('./atad/train.csv')  # df.shape is  (7377418, 6)   df.target 1 ratio is  0.5035170841614234
print('train_df.shape is ', train_df.shape)
print('train_df.head(10) is ', train_df.head(10))
print('train_df.target 1 ratio is ', train_df.target.sum()/len(train_df))
print('train_df.source_system_tab nunique is ', train_df['source_system_tab'].nunique())
print('train_df.source_screen_name nunique is ', train_df['source_screen_name'].nunique())
print('train_df.source_type nunique is ', train_df['source_type'].nunique())
#source_system_tab,source_screen_name,source_type


#test_df = pd.read_csv('./atad/test.csv')
#print('test_df.shape is ', test_df.shape)

members_df = pd.read_csv('./atad/members.csv')
print('members_df.shape is ', members_df.shape)
print('members_df.head(10) is ', members_df.head(10))

song_df = pd.read_csv('./atad/songs.csv')
print('song_df.shape is ', song_df.shape)
print('song_df.head(10) is ', song_df.head(10))

song_extra_info_df = pd.read_csv('./atad/song_extra_info.csv')
print('song_extra_info_df.shape is ', song_extra_info_df.shape)
print('song_extra_info_df.head(10) is ', song_extra_info_df.head(10))

#print('hello world!')


