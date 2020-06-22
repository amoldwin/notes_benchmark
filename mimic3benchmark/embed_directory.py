import numpy as np
import argparse
import sys 
import pickle
from util import *
from nltk.tokenize import sent_tokenize, word_tokenize
import psutil
import os
process = psutil.Process(os.getpid())
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

d2v_dim = 300


embedded_path = 'NEW_length-of-stay_embedded_' + str(d2v_dim)
vec_file = 'doc2vec_length-of-stay_' + str(d2v_dim) + '.pk'
# embedded_path = 'length-of-stay_embedded_' + str(d2v_dim) + '/train/'

train_dir = './data/length-of-stay/train/'

columns = []
corpus = []
docindex = 0
for filename in os.listdir(train_dir):
    sentences = []
    with open(os.path.join(train_dir, filename)) as tsfile:
        ts = dataframe_from_csv(tsfile, index_col=None)

        # print(ts.head(3))
        if 'TEXT' in ts.columns:
            columns = ts.columns 
            # break
            ts = ts[ts.TEXT.notna()]

            ts.apply(lambda row: corpus.append(TaggedDocument(word_tokenize(row.TEXT),[len(corpus)])), axis = 1)
# with open('d2vcorpus_length-of-stay', 'wb') as fin:
#     pickle.dump(corpus, fin)

# corpus = pickle.load(open('d2vcorpus_DEC', "rb")) 
X = Doc2Vec(corpus, vector_size=d2v_dim, window=2, min_count=15, workers=12, iter = 15, max_vocab_size=10000)
with open(vec_file, 'wb') as fin:
    pickle.dump(X, fin)
print('finished training doc2vec')       

X = pickle.load(open(vec_file, "rb")) 
print('Loaded Doc2vec')


    
embed_dir = os.path.join('./data/', embedded_path)
try:
    os.mkdir(embed_dir)
    os.mkdir(os.path.join(embed_dir,'train'))
    os.mkdir(os.path.join(embed_dir,'test'))
except:
    print('mkdir failed')
embed_dir = os.path.join(embed_dir,'train')
# except OSError:
#     print ("Creation of the directory %s failed" % os.path.join('./data/', embedded_path))


def create_row(row,X):
    if row.TEXT=='':
        return ','.join(str(v) for v in (row.values.tolist()[:txtindex] + row.values.tolist()[txtindex+1 :] + 300*['']))
    else:
        return ','.join(str(v) for v in (row.values.tolist()[:txtindex] + row.values.tolist()[txtindex+1 :] + X.infer_vector(word_tokenize((row.TEXT))).tolist()))
for filename in os.listdir(train_dir):
    with open(os.path.join(train_dir, filename)) as tsfile:
        if filename == 'listfile.csv':
            with open(os.path.join(embed_dir, filename), 'w') as file:
                for line in tsfile:
                    file.write(line)
                continue
        columns = [i for i in columns]
        ret = columns[0]
        for i in columns[1:]: 
            if not i=='TEXT':
                ret = ret + ',' +  i 
        for i in range(0,d2v_dim): ret = ret + ',' +  str(i)
        # ret = ret + ',' + 'was_text'
        ret = [ret]
        ts = dataframe_from_csv(tsfile, index_col=None).fillna('')
        if 'TEXT' in ts.columns:
            #ts.apply(lambda row: print('VALUES',row.values.tolist() ), axis = 1)
            # print('VALUES',X.infer_vector(word_tokenize('')))
            txtindex = columns.index('TEXT')
            ts.apply(lambda row: ret.append(create_row(row,X)), axis = 1)
            with open(os.path.join(embed_dir, filename), 'w') as file:
                for line in ret:
                    print(line)
                    file.write(line+'\n')





train_dir = './data/length-of-stay/test/'
embed_dir = os.path.join('./data/', embedded_path)
embed_dir = os.path.join(embed_dir,'test')

for filename in os.listdir(train_dir):
    with open(os.path.join(train_dir, filename)) as tsfile:
        if filename == 'listfile.csv':
            with open(os.path.join(embed_dir, filename), 'w') as file:
                for line in tsfile:
                    file.write(line)
                continue
        columns = [i for i in columns]
        ret = columns[0]
        for i in columns[1:]: 
            if not i=='TEXT':
                ret = ret + ',' +  i 
        for i in range(0,d2v_dim): ret = ret + ',' +  str(i)
        # ret = ret + ',' + 'was_text'
        ret = [ret]
        ts = dataframe_from_csv(tsfile, index_col=None).fillna('')
        if 'TEXT' in ts.columns:
            #ts.apply(lambda row: print('VALUES',row.values.tolist() ), axis = 1)
            # print('VALUES',X.infer_vector(word_tokenize('')))
            txtindex = columns.index('TEXT')
            ts.apply(lambda row: ret.append(create_row(row,X)), axis = 1)
            with open(os.path.join(embed_dir, filename), 'w') as file:
                for line in ret:
                    print(line)
                    file.write(line+'\n')
     