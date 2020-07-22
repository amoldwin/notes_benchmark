from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import json
import random
import pandas as pd

from mimic3models.feature_extractor import extract_features

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


def convert_to_dict(data, header, channel_info):
    """ convert data from readers output in to array of arrays format """
    ret = [[] for i in range(data.shape[1] - 1)]
    for i in range(1, data.shape[1]):
        ret[i-1] = [(t, x) for (t, x) in zip(data[:, 0], data[:, i]) if x != ""]
        channel = header[i]
        if len(channel_info[channel]['possible_values']) != 0:
            ret[i-1] = list(map(lambda x: (x[0], channel_info[channel]['values'][x[1]]), ret[i-1]))
        ret[i-1] = list(map(lambda x: (float(x[0]), float(x[1])), ret[i-1]))
    return ret


def extract_features_from_rawdata(chunk, header, period, features):
    with open(os.path.join(os.path.dirname(__file__), "resources/channel_info.json")) as channel_info_file:
        channel_info = json.loads(channel_info_file.read())
    data = [convert_to_dict(X, header, channel_info) for X in chunk]
    return extract_features(data, period, features)


def read_chunk(reader, chunk_size):
    data = {}
    for i in range(chunk_size):
        ret = reader.read_next()
        for k, v in ret.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    data["header"] = data["header"][0]
    return data


def sort_and_shuffle(data, batch_size):
    """ Sort data by the length and then make batches and shuffle them.
        data is tuple (X1, X2, ..., Xn) all of them have the same length.
        Usually data = (X, y).
    """
    assert len(data) >= 2
    data = list(zip(*data))

    random.shuffle(data)

    old_size = len(data)
    rem = old_size % batch_size
    head = data[:old_size - rem]
    tail = data[old_size - rem:]
    data = []

    head.sort(key=(lambda x: x[0].shape[0]))

    mas = [head[i: i+batch_size] for i in range(0, len(head), batch_size)]
    random.shuffle(mas)

    for x in mas:
        data += x
    data += tail

    data = list(zip(*data))
    return data


def add_common_arguments(parser):
    """ Add all the parameters which are common across the tasks
    """
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--dim', type=int, default=256,
                        help='number of hidden units')
    parser.add_argument('--depth', type=int, default=1,
                        help='number of bi-LSTMs')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of chunks to train')
    parser.add_argument('--load_state', type=str, default="",
                        help='state file path')
    parser.add_argument('--mode', type=str, default="train",
                        help='mode: train or test')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--l2', type=float, default=0, help='L2 regularization')
    parser.add_argument('--l1', type=float, default=0, help='L1 regularization')
    parser.add_argument('--save_every', type=int, default=1,
                        help='save state every x epoch')
    parser.add_argument('--prefix', type=str, default="",
                        help='optional prefix of network name')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--rec_dropout', type=float, default=0.0,
                        help="dropout rate for recurrent connections")
    parser.add_argument('--batch_norm', type=bool, default=False,
                        help='batch normalization')
    parser.add_argument('--timestep', type=float, default=1.0,
                        help="fixed timestep used in the dataset")
    parser.add_argument('--imputation', type=str, default='previous')
    parser.add_argument('--small_part', dest='small_part', action='store_true')
    parser.add_argument('--whole_data', dest='small_part', action='store_false')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--beta_1', type=float, default=0.9,
                        help='beta_1 param for Adam optimizer')
    parser.add_argument('--verbose', type=int, default=2)
    parser.add_argument('--size_coef', type=float, default=4.0)
    parser.add_argument('--normalizer_state', type=str, default=None,
                        help='Path to a state file of a normalizer. Leave none if you want to '
                             'use one of the provided ones.')
    parser.set_defaults(small_part=False)


class DeepSupervisionDataLoader:
    r"""
    Data loader for decompensation and length of stay task.
    Reads all the data for one patient at once.

    Parameters
    ----------
    dataset_dir : str
        Directory where timeseries files are stored.
    listfile : str
        Path to a listfile. If this parameter is left `None` then
        `dataset_dir/listfile.csv` will be used.
    """
    def __init__(self, dataset_dir, listfile=None, small_part=False, sources=[], timesteps=None, condensed=False):

        self._dataset_dir = dataset_dir
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()[1:]  # skip the header

        self._data = [line.split(',') for line in self._data]
        self._data = [(x, float(t), y) for (x, t, y) in self._data]
        self._data = sorted(self._data)

        self._sources = sources
        self._timesteps = timesteps
        self._condensed=condensed


        mas = {"X": [],
               "ts": [],
               "ys": [],
               "name": []}
        i = 0
        while i < len(self._data):
            j = i
            cur_stay = self._data[i][0]
            cur_ts = []
            cur_labels = []
            while j < len(self._data) and self._data[j][0] == cur_stay:
                cur_ts.append(self._data[j][1])
                cur_labels.append(self._data[j][2])
                j += 1

            cur_X, header = self._read_timeseries(cur_stay)
            mas["X"].append(cur_X)
            mas["ts"].append(cur_ts)
            mas["ys"].append(cur_labels)
            mas["name"].append(cur_stay)

            i = j
            if small_part and len(mas["name"]) == 256:
                break

        self._data = mas

    def _read_timeseries(self, ts_filename, time_bound=None):
        ret = []
        # print('filename', ts_filename)
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            ts_df = pd.read_csv(tsfile, dtype='str').fillna('') 
        
        ts_df = ts_df.replace('',np.nan)
        not_in_structured = ['Hours', 'WORDS', 'CUIS', 'DOC2VEC','TEXT']
        in_structured = [col for col in ts_df.columns if col not in not_in_structured]
        
        if self._condensed:
            has_text = ts_df['TEXT'].notnull()
            ts_df[in_structured] = ts_df[in_structured].fillna(method='ffill')
            ts_df[in_structured]  = ts_df[in_structured].fillna(method='bfill')[has_text]

            # ts_df = ts_df.dropna(axis=0,how='all',subset=['TEXT'])
        ts_df = ts_df.replace(np.nan,'')

        
        included_columns = ['Hours']
        if 'structured_data' in self._sources:
            not_in_structured = ['Hours', 'WORDS', 'CUIS', 'DOC2VEC','TEXT']
            in_structured = [col for col in ts_df.columns if col not in not_in_structured]
            included_columns = included_columns + in_structured
            ts_df['Glascow coma scale total'] =ts_df['Glascow coma scale total'].apply(lambda x: str(int(float(x))) if not x=='' else '' ) 
            ts_df['Glucose'] =ts_df['Glucose'].apply(lambda x: str(int(float(x))) if not x=='' else '' )#.apply(lambda x: '' if not x.isnumeric() else x ) 

        if 'doc2vec' in self._sources:
            included_columns = included_columns + ['DOC2VEC']
        if 'words' in self._sources:
            included_columns = included_columns + ['WORDS']

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
        

        new_df = new_df.replace('',np.nan)
        new_df = new_df.dropna(axis=0,how='all',subset=[col for col in new_df.columns if not col=='Hours'])
        # new_df.to_csv('test_reader.csv')
        
        if not time_bound==None:
            time_filter = new_df['Hours'].apply(lambda x: float(x)<=float(time_bound+1e-6))
            new_df=new_df[time_filter]
        header = new_df.columns.to_list()
        

        new_df = new_df.replace(np.nan,'')
        
        ret = [np.array(row) for row in new_df.values.tolist()]

        if ret ==[]:
            # ret = [['']*len(included_columns)]
            assert False, (ts_filename,new_df)
            # ret = [[0.0]*len(header)]
            print(ts_filename, ' had no  events')
        assert header[0] == "Hours"
        # for row in ret:
        #     if 'NEG' in row:
        #         print('filename',ts_filename)
        #         print([(header[i],row[i]) for i in range(len(row))])
        # for line in tsfile:
        #     mas = line.strip().split(',')
        #     ret.append(np.array(mas))
        return (np.stack(ret), header)

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def pad_zeros(arr, min_length=None):
    """
    `arr` is an array of `np.array`s

    The function appends zeros to every `np.array` in `arr`
    to equalize their first axis lenghts.
    """
    dtype = arr[0].dtype
    max_len = max([x.shape[0] for x in arr])
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret)
