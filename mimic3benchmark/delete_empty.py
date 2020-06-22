import numpy as np
import argparse
import sys 
import pickle
from util import *
import psutil
import os
process = psutil.Process(os.getpid())
import logging
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default = None)
args = parser.parse_args()
print('started', args.task)

new_dir = args.task
new_train_dir = new_dir + 'train/'
new_test_dir = new_dir + 'test/'


old_train_listfile_path = new_train_dir+'listfile.csv'
old_test_listfile_path = new_test_dir+'listfile.csv'

new_train_listfile_path = new_train_dir+'new_listfile.csv'
new_test_listfile_path = new_test_dir+'new_listfile.csv'





def filter_list(path, new_path, dir):
    with open(path, 'r') as listfile:
        lines = listfile.readlines()
    orig_lines = lines
    lines = [line.split(',') for line in lines]
    with open(new_path, 'w') as newfile:

        newfile.write(orig_lines[0])

        for i in range(1,len(lines)):
            num_lines = sum(1 for line in open(dir+lines[i][0]))
            if num_lines <2:
                print(lines[i][0])
                print(num_lines)
            else:
                newfile.write(orig_lines[i])

        
filter_list(old_test_listfile_path, new_test_listfile_path, new_test_dir)
filter_list(old_train_listfile_path, new_train_listfile_path, new_train_dir)
