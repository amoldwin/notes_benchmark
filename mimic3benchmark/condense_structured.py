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


old_path = './data/condensed_'+args.task+'/'
old_train_dir = old_path + 'train'
old_test_dir = old_path + 'test'

new_dir = './data/structured_condensed_'+args.task+'/'
new_train_dir = new_dir + 'train/'
new_test_dir = new_dir + 'test/'

try:
    os.mkdir(new_dir)
    os.mkdir(new_train_dir)
    os.mkdir(new_test_dir)

except:
    print('mkdir failed')


def make_structured_dir(dir, out_dir):
    for filename in os.listdir(dir):
        with open(dir+'/'+filename) as tsfile:

            if filename == 'listfile.csv':
                with open(out_dir +'/'+ filename, 'w') as file:
                    for line in tsfile:
                        file.write(line)
                    continue
            ts = dataframe_from_csv(tsfile, index_col=None)#.fillna('')
            # new_ts = ts.apply(lambda row: find_previous(row), axis = 1)

            columns = [i for i in ts.columns]

            # timefilter= np.isnan(ts['288'])
            
            new_ts = ts.iloc[:,0:18]
            new_ts.to_csv(out_dir+'/'+filename, index = False)

            # for i in range(1,18):
            #     ts[ts[columns[i]]==""][columns[i]]= np.NaN




            # just_note_times = ts[timefilter]
            # num_left = len(ts.index)
            # prev_vals = [num_left*[]]
            # print(ts.head(10))
            # print(ts['Hours'].head(3))
            # print(columns)

print('starting test data')
make_structured_dir(old_test_dir,new_test_dir )
print('starting train data')
make_structured_dir(old_train_dir,new_train_dir )



# columns = []
# corpus = []
# docindex = 0
# for filename in os.listdir(train_dir):
#     sentences = []
#     with open(os.path.join(train_dir, filename)) as tsfile:
#         ts = dataframe_from_csv(tsfile, index_col=None)

#         # print(ts.head(3))
#         if 'TEXT' in ts.columns:
#             columns = ts.columns 
#             # break
#             ts = ts[ts.TEXT.notna()]

#             ts.apply(lambda row: corpus.append(TaggedDocument(word_tokenize(row.TEXT),[len(corpus)])), axis = 1)
# # with open('d2vcorpus_'+args.task+'', 'wb') as fin:
# #     pickle.dump(corpus, fin)

# # corpus = pickle.load(open('d2vcorpus_DEC', "rb")) 
# X = Doc2Vec(corpus, vector_size=d2v_dim, window=2, min_count=15, workers=12, iter = 15, max_vocab_size=10000)
# with open(vec_file, 'wb') as fin:
#     pickle.dump(X, fin)
# print('finished training doc2vec')       

# X = pickle.load(open(vec_file, "rb")) 
# print('Loaded Doc2vec')


    
# embed_dir = os.path.join('./data/', embedded_path)
# try:
#     os.mkdir(embed_dir)
#     os.mkdir(os.path.join(embed_dir,'train'))
#     os.mkdir(os.path.join(embed_dir,'test'))
# except:
#     print('mkdir failed')
# embed_dir = os.path.join(embed_dir,'train')
# # except OSError:
# #     print ("Creation of the directory %s failed" % os.path.join('./data/', embedded_path))


# def create_row(row,X):
#     if row.TEXT=='':
#         return ','.join(str(v) for v in (row.values.tolist()[:txtindex] + row.values.tolist()[txtindex+1 :] + 300*['']))
#     else:
#         return ','.join(str(v) for v in (row.values.tolist()[:txtindex] + row.values.tolist()[txtindex+1 :] + X.infer_vector(word_tokenize((row.TEXT))).tolist()))
# for filename in os.listdir(train_dir):
#     with open(os.path.join(train_dir, filename)) as tsfile:
#         if filename == 'listfile.csv':
#             with open(os.path.join(embed_dir, filename), 'w') as file:
#                 for line in tsfile:
#                     file.write(line)
#                 continue
#         columns = [i for i in columns]
#         ret = columns[0]
#         for i in columns[1:]: 
#             if not i=='TEXT':
#                 ret = ret + ',' +  i 
#         for i in range(0,d2v_dim): ret = ret + ',' +  str(i)
#         # ret = ret + ',' + 'was_text'
#         ret = [ret]
#         ts = dataframe_from_csv(tsfile, index_col=None).fillna('')
#         if 'TEXT' in ts.columns:
#             #ts.apply(lambda row: print('VALUES',row.values.tolist() ), axis = 1)
#             # print('VALUES',X.infer_vector(word_tokenize('')))
#             txtindex = columns.index('TEXT')
#             ts.apply(lambda row: ret.append(create_row(row,X)), axis = 1)
#             with open(os.path.join(embed_dir, filename), 'w') as file:
#                 for line in ret:
#                     print(line)
#                     file.write(line+'\n')





# train_dir = './data/'+args.task+'/test/'
# embed_dir = os.path.join('./data/', embedded_path)
# embed_dir = os.path.join(embed_dir,'test')

# for filename in os.listdir(train_dir):
#     with open(os.path.join(train_dir, filename)) as tsfile:
#         if filename == 'listfile.csv':
#             with open(os.path.join(embed_dir, filename), 'w') as file:
#                 for line in tsfile:
#                     file.write(line)
#                 continue
#         columns = [i for i in columns]
#         ret = columns[0]
#         for i in columns[1:]: 
#             if not i=='TEXT':
#                 ret = ret + ',' +  i 
#         for i in range(0,d2v_dim): ret = ret + ',' +  str(i)
#         # ret = ret + ',' + 'was_text'
#         ret = [ret]
#         ts = dataframe_from_csv(tsfile, index_col=None).fillna('')
#         if 'TEXT' in ts.columns:
#             #ts.apply(lambda row: print('VALUES',row.values.tolist() ), axis = 1)
#             # print('VALUES',X.infer_vector(word_tokenize('')))
#             txtindex = columns.index('TEXT')
#             ts.apply(lambda row: ret.append(create_row(row,X)), axis = 1)
#             with open(os.path.join(embed_dir, filename), 'w') as file:
#                 for line in ret:
#                     print(line)
#                     file.write(line+'\n')
     