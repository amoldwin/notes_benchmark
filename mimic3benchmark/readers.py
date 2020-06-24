from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import random
import pandas as pd
import json


DOC2VEC_DIM = 300
cui_vocab_path = 'jupyter_notebooks/cui_sorted_id.json'
words_vocab_path = 'jupyter_notebooks/words_sorted_id.json'
with open(cui_vocab_path, 'r') as vocfile:
    cuis_vocab = json.load(vocfile)
with open(words_vocab_path, 'r') as vocfile:
    words_vocab = json.load(vocfile)
max_len=120
def encode_cuis(doc):
    if doc=='nan' or doc=='':
        return ''
    nparr = [cuis_vocab[word] if word in cuis_vocab else 0 for word in doc.split(' ')]
    if len(nparr)>max_len:
        nparr=nparr[:max_len]
    else:
        nparr = nparr + [0]*(max_len-len(nparr))
    return nparr
def encode_words(doc):
    if doc=='nan' or doc=='':
        return ''
    nparr = [words_vocab[word] if word in words_vocab else 0 for word in doc.split(' ')]
    if len(nparr)>max_len:
        nparr=nparr[:max_len]
    else:
        nparr = nparr + [0]*(max_len-len(nparr))
    return nparr


def read_timeseries(self, ts_filename, time_bound=None):
    ret = []
    # print('filename', ts_filename)
    with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
        ts_df = pd.read_csv(tsfile, dtype='str').fillna('') 

    ts_df = ts_df.replace('',np.nan)
    if self._condensed:
        has_text = ts_df['TEXT'].notnull()
        ts_df = ts_df.fillna(method='ffill')[has_text]
        # ts_df = ts_df.dropna(axis=0,how='all',subset=['TEXT'])
        # assert False, ts_df
    ts_df = ts_df.replace(np.nan,'')




    included_columns = ['Hours']
    if 'structured_data' in self._sources:
        not_in_structured = ['Hours', 'WORDS', 'CUIS', 'DOC2VEC','TEXT']
        in_structured = [col for col in ts_df.columns if col not in not_in_structured]
        included_columns = included_columns + in_structured
        ts_df['Glascow coma scale total'] =ts_df['Glascow coma scale total'].apply(lambda x: str(int(float(x))) if not x=='' else '' ) 
        ts_df['Glucose'] =ts_df['Glucose'].apply(lambda x: '' if not x.isnumeric() else x ) 

    if 'doc2vec' in self._sources:
        included_columns = included_columns + ['DOC2VEC']
    if 'words' in self._sources:
        included_columns = included_columns + ['WORDS']
        # assert False, ts_df['WORDS']

        ts_df['WORDS'] = ts_df['WORDS'].astype(str).apply(lambda x: encode_words(x))
    if 'cuis' in self._sources:
        included_columns = included_columns + ['CUIS']
        ts_df['CUIS'] = ts_df['CUIS'].apply(lambda x: encode_cuis(x))
    if 'cuis' in self._sources or 'words' in self._sources:
        assert len(included_columns)==2, 'Cannot combine bag of words/concepts with other features'
    # header = tsfile.readline().strip().split(',')

    new_df = ts_df[included_columns]
    if 'DOC2VEC' in included_columns:
        for i in range(0,DOC2VEC_DIM):
            new_df['DOC2VEC'] = new_df['DOC2VEC'].apply(lambda x: [float(y) for y in str(x).replace('[','').replace(']','').split(',')] if not (x=='') else '')
            new_df['d2v->'+str(i)] = new_df['DOC2VEC'].apply(lambda x: x[i] if not x=='' else '')
        new_df = new_df[[col for col in new_df.columns if not col=='DOC2VEC']]
    if 'WORDS' in included_columns:
        for i in range(0,max_len):
            new_df['WORDS'] = new_df['WORDS'].apply(lambda x: [float(y) for y in str(x).replace('[','').replace(']','').split(',')] if not (x=='') else '')
            new_df['words->'+str(i)] = new_df['WORDS'].apply(lambda x: x[i] if not x=='' else np.nan)
        new_df = new_df[[col for col in new_df.columns if not col=='WORDS']]
    if 'CUIS' in included_columns:
        for i in range(0,max_len):
            new_df['CUIS'] = new_df['CUIS'].apply(lambda x: [float(y) for y in str(x).replace('[','').replace(']','').split(',')] if not (x=='') else '')
            new_df['cuis->'+str(i)] = new_df['CUIS'].apply(lambda x: x[i] if not x=='' else np.nan)
        new_df = new_df[[col for col in new_df.columns if not col=='CUIS']]
    
    new_df = new_df.dropna(axis=0,how='all',subset=[col for col in new_df.columns if not col=='Hours'])
    
    
    # assert False, new_df
    if not time_bound==None:
        time_filter = new_df['Hours'].apply(lambda x: x<=(time_bound+1e-6))
        new_df=new_df[time_filter]
    header = new_df.columns.to_list()
    # assert False, header
    ret = [np.array(row) for row in new_df.values.tolist()]


    if ret ==[]:
        assert False
        # ret = [[0.0]*len(header)]
    # assert False, header
    assert header[0] == "Hours"
    # for row in ret:
    #     if 'NEG' in row:
    #         print('filename',ts_filename)
    #         print([(header[i],row[i]) for i in range(len(row))])
    # for line in tsfile:
    #     mas = line.strip().split(',')
    #     ret.append(np.array(mas))
    return (np.stack(ret), header)



class Reader(object):
    def __init__(self, dataset_dir, listfile=None):
        self._dataset_dir = dataset_dir
        self._current_index = 0
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self._data = self._data[1:]

    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)
    
   


class DecompensationReader(Reader):
    def __init__(self, dataset_dir, listfile=None, sources=[], timesteps=None, condensed=False):
        """ Reader for decompensation prediction task.
        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, float(t), int(y)) for (x, t, y) in self._data]
        self._sources = sources
        self._timesteps = timesteps
        self._condensed=condensed


    def read_example(self, index):
        """ Read the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Directory with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : int (0 or 1)
                Mortality within next 24 hours.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of examples (exclusive).")

        name = self._data[index][0]
        t = self._data[index][1]
        y = self._data[index][2]
        (X, header) = read_timeseries(self, name, t)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}


class InHospitalMortalityReader(Reader):
    def __init__(self, dataset_dir, listfile=None, period_length=48.0, sources=[], timesteps=None, condensed=False):
        """ Reader for in-hospital moratality prediction task.

        :param dataset_dir:   Directory where timeseries files are stored.
        :param listfile:      Path to a listfile. If this parameter is left `None` then
                              `dataset_dir/listfile.csv` will be used.
        :param period_length: Length of the period (in hours) from which the prediction is done.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, int(y)) for (x, y) in self._data]
        self._period_length = period_length
        self._sources = sources
        self._timesteps = timesteps
        self._condensed=condensed
    
    def read_example(self, index):

        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : int (0 or 1)
                In-hospital mortality.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")
        name = self._data[index][0]
        t = self._period_length
        y = self._data[index][1]
        (X, header) = read_timeseries(self, name)
        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}


class LengthOfStayReader(Reader):
    def __init__(self, dataset_dir, listfile=None, sources=[], timesteps=None, condensed=False):
        """ Reader for length of stay prediction task.

        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, float(t), float(y)) for (x, t, y) in self._data]
        self._sources = sources
        self._timesteps = timesteps
        self._condensed=condensed


    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : float
                Remaining time in ICU.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._data[index][1]
        y = self._data[index][2]
        (X, header) = read_timeseries(self,name, t)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}


class PhenotypingReader(Reader):
    def __init__(self, dataset_dir, listfile=None, sources=[], timesteps=None, condensed=False):
        """ Reader for phenotype classification task.

        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(mas[0], float(mas[1]), list(map(int, mas[2:]))) for mas in self._data]
        self._sources = sources
        self._timesteps = timesteps
        self._condensed=condensed
   
    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : array of ints
                Phenotype labels.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._data[index][1]
        y = self._data[index][2]
        (X, header) = read_timeseries(self,name)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}


class MultitaskReader(Reader):
    def __init__(self, dataset_dir, listfile=None, sources=[], timesteps=None, condensed=False):
        """ Reader for multitask learning.

        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._condensed=condensed
        def process_ihm(x):
            return list(map(int, x.split(';')))

        def process_los(x):
            x = x.split(';')
            if x[0] == '':
                return ([], [])
            return (list(map(int, x[:len(x)//2])), list(map(float, x[len(x)//2:])))

        def process_ph(x):
            return list(map(int, x.split(';')))

        def process_decomp(x):
            x = x.split(';')
            if x[0] == '':
                return ([], [])
            return (list(map(int, x[:len(x)//2])), list(map(int, x[len(x)//2:])))

        self._data = [(fname, float(t), process_ihm(ihm), process_los(los),
                       process_ph(pheno), process_decomp(decomp))
                      for fname, t, ihm, los, pheno, decomp in self._data]
        self._sources = sources
        self._timesteps = timesteps
    

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Return dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            ihm : array
                Array of 3 integers: [pos, mask, label].
            los : array
                Array of 2 arrays: [masks, labels].
            pheno : array
                Array of 25 binary integers (phenotype labels).
            decomp : array
                Array of 2 arrays: [masks, labels].
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        (X, header) = read_timeseries(self,name)

        return {"X": X,
                "t": self._data[index][1],
                "ihm": self._data[index][2],
                "los": self._data[index][3],
                "pheno": self._data[index][4],
                "decomp": self._data[index][5],
                "header": header,
                "name": name}
