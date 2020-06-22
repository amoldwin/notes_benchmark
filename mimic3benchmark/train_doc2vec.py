import pandas as pd
import os
import numpy as np
import pickle
import argparse 
parser = argparse.ArgumentParser()

parser.add_argument('--test', action='store_true')
parser.add_argument('--data',type=str, default='../../LHC_mimic/mimic3_1.4/raw_data/NOTEEVENTS.csv')
parser.add_argument('--output_dir', type=str, default='../resources/')
args = parser.parse_args()


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import sent_tokenize, word_tokenize
import logging

doc2vec_model_path = 'resources/doc2vec_allnotes.pk'
notespath = args.data
if args.test==True:
    notes_df  = pd.read_csv(notespath, nrows=100)
else:
    notes_df  = pd.read_csv(notespath)
d2v_dim = 300







corpus = notes_df['TEXT'].tolist()
corpus = [TaggedDocument(word_tokenize(txt),[i]) for i,txt in enumerate(corpus)]

X = Doc2Vec(corpus, alpha = 0.025, vector_size=d2v_dim, window=4, min_count=4, workers=12, iter = 100, max_vocab_size=10000)    

with open(doc2vec_model_path, 'wb') as fin:
    pickle.dump(X, fin)