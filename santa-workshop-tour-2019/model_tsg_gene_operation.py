import os
import time
import sys
import random
import numpy as np
import pandas as pd
sys.setrecursionlimit(10000) 

start_t_global = time.time()

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

files_lst = get_all_files('./submission/min/')
print('files_lst is ', files_lst)

print('len of files_lst is ', len(files_lst))

print('total cost time ', time.time() - start_t_global)
