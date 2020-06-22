import sys, argparse, csv
import pandas
import re
from util import *


path = '../data/in-hospital-mortality/val_listfile.csv'


with open(path, 'r') as file:
    lf = dataframe_from_csv(file)
counts = lf['y_true'].value_counts().tolist()
print(counts)
is_neg = lf['y_true']==0
is_pos = lf['y_true']==1
# print(lf[is_neg].head(3))

def switch(num):
    num=float(num)
    if num==0:
        num=1
    elif num==1:
            num=0
    return num

pos_exs = lf[is_pos].sample(n=counts[1])
# neg_exs.reset_index(inplace=True)
ratio = counts[1]/counts[0]
neg_exs = lf[is_neg].sample(n=counts[0])
both = pos_exs.append(neg_exs).sample(frac=1)
both['y_true']=both['y_true'].apply(lambda item: switch(item))
print(both)
both.to_csv('../data/in-hospital-mortality/switched_val.csv')

# print('ratio was',ratio)
# for i in range(0,counts[1]):
#     ind = rand    