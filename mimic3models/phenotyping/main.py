from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import imp
import re

from mimic3models.phenotyping import utils
from mimic3benchmark.readers import PhenotypingReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

from keras.callbacks import ModelCheckpoint, CSVLogger

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of phenotyping task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/phenotyping/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')

parser.add_argument('--doc2vec', action='store_true')
parser.add_argument('--cuis', action='store_true')
parser.add_argument('--words', action='store_true')
parser.add_argument('--structured_data', action='store_true')
parser.add_argument('--timesteps', type=str, default='both')
parser.add_argument('--weighted', action='store_true')
parser.add_argument('--condensed', action='store_true')


args = parser.parse_args()

print(args)
vocab_size=None
embedding=False
sources = []
experiment_name='PHE_'
if args.doc2vec:
    sources.append('doc2vec')
    experiment_name=experiment_name+'doc2vec_'
if args.cuis:
    sources.append('cuis')
    experiment_name=experiment_name+'cuis_'
    embedding=True
    vocab_size=51893
if args.words:
    sources.append('words')
    experiment_name=experiment_name+'words_'
    embedding=True
    vocab_size=1891434
if args.structured_data:
    sources.append('structured_data')
    experiment_name=experiment_name+'structured_'
if args.weighted:
    experiment_name=experiment_name+'weighted_'
if args.condensed:
    experiment_name=experiment_name+'condensed_'



if args.small_part:
    args.save_every = 2**30

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

# Build readers, discretizers, normalizers
train_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                 listfile=os.path.join(args.data, 'train_listfile.csv'), sources=sources, timesteps=args.timesteps, condensed=args.condensed)

val_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                               listfile=os.path.join(args.data, 'val_listfile.csv'), sources=sources, timesteps=args.timesteps, condensed=args.condensed)
reader_header = train_reader.read_example(0)['header']

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero', header = reader_header, sources = sources)


discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]






normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'ph_ts{}.input_str_previous.start_time_zero.normalizer'.format(args.timestep)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

train_data_gen = utils.BatchGen(train_reader, discretizer,
                                normalizer, args.batch_size,
                                args.small_part, target_repl, shuffle=True)

n_bins = len(train_data_gen.next()[0][0])
print('n_bins', n_bins)
print('n_bins', len(train_data_gen.next()[0][0]))
print('n_bins', len(train_data_gen.next()[0][0]))
print('n_bins', len(train_data_gen.next()[0][0]))





args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'ph'
args_dict['num_classes'] = 25
args_dict['target_repl'] = target_repl
args_dict['input_dim']= len(discretizer_header)
args_dict['embedding'] = embedding
args_dict['vocab_size'] = vocab_size
args_dict['embed_dim'] = 16
args_dict['n_bins'] = n_bins
args_dict['seq_length'] = 120

# Build the model
print("==> using model {}".format(args.network))
model_module = imp.load_source(os.path.basename(args.network), args.network)
model = model_module.Network(**args_dict)
suffix = ".bs{}{}{}.ts{}{}".format(args.batch_size,
                                   ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                   ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                   args.timestep,
                                   ".trc{}".format(args.target_repl_coef) if args.target_repl_coef > 0 else "")
model.final_name = args.prefix + model.say_name() + suffix
print("==> model.final_name:", model.final_name)


# Compile the model
print("==> compiling the model")
optimizer_config = {'class_name': args.optimizer,
                    'config': {'lr': args.lr,
                               'beta_1': args.beta_1}}

# NOTE: one can use binary_crossentropy even for (B, T, C) shape.
#       It will calculate binary_crossentropies for each class
#       and then take the mean over axis=-1. Tre results is (B, T).
if target_repl:
    loss = ['binary_crossentropy'] * 2
    loss_weights = [1 - args.target_repl_coef, args.target_repl_coef]
else:
    loss = 'binary_crossentropy'
    loss_weights = None

model.compile(optimizer=optimizer_config,
              loss=loss,
              loss_weights=loss_weights)
model.summary()

# Load model weights
n_trained_chunks = 0
state_to_test = args.load_state
if args.load_state != "":
    model.load_weights(args.load_state)
    n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state).group(1))
elif args.mode == 'test':
    state_to_test = os.path.join(args.output_dir, 'keras_states/' + model.final_name +experiment_name+ '.state')
    model.load_weights(state_to_test)

# Build data generators
train_data_gen = utils.BatchGen(train_reader, discretizer,
                                normalizer, args.batch_size,
                                args.small_part, target_repl, shuffle=True)
val_data_gen = utils.BatchGen(val_reader, discretizer,
                              normalizer, args.batch_size,
                              args.small_part, target_repl, shuffle=False)


if args.mode == 'train':
    # Prepare training
    path = os.path.join(args.output_dir, 'keras_states/' + model.final_name +experiment_name+ '.state')

    metrics_callback = keras_utils.PhenotypingMetrics(train_data_gen=train_data_gen,
                                                      val_data_gen=val_data_gen,
                                                      batch_size=args.batch_size,
                                                      verbose=args.verbose)
    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, save_best_only=True, monitor='val_ave_mcc', mode='max')

    keras_logs = os.path.join(args.output_dir, 'keras_logs')
    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name +experiment_name+ '.csv'),
                           append=True, separator=';')

    print("==> training")
    model.fit_generator(generator=train_data_gen,
                        steps_per_epoch=train_data_gen.steps,
                        validation_data=val_data_gen,
                        validation_steps=val_data_gen.steps,
                        epochs=n_trained_chunks + args.epochs,
                        initial_epoch=n_trained_chunks,
                        callbacks=[metrics_callback, saver, csv_logger],
                        verbose=args.verbose)

elif args.mode == 'test':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_data_gen
    del val_data_gen

    test_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'test'),
                                    listfile=os.path.join(args.data, 'test_listfile.csv'), sources=sources, timesteps=args.timesteps, condensed=args.condensed)

    test_data_gen = utils.BatchGen(test_reader, discretizer,
                                   normalizer, args.batch_size,
                                   args.small_part, target_repl,
                                   shuffle=False, return_names=True)

    names = []
    ts = []
    labels = []
    predictions = []
    for i in range(test_data_gen.steps):
        print("predicting {} / {}".format(i, test_data_gen.steps), end='\r')
        ret = next(test_data_gen)
        x = ret["data"][0]
        y = ret["data"][1]
        cur_names = ret["names"]
        cur_ts = ret["ts"]
        x = np.array(x)
        pred = model.predict_on_batch(x)
        predictions += list(pred)
        labels += list(y)
        names += list(cur_names)
        ts += list(cur_ts)

    metrics.print_metrics_multilabel(labels, predictions)
    path = os.path.join(args.output_dir, "test_predictions", os.path.basename(state_to_test)) +experiment_name+ ".csv"
    utils.save_results(names, ts, predictions, labels, path)

else:
    raise ValueError("Wrong value for args.mode")
