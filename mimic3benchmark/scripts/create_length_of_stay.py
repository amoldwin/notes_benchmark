from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import numpy as np
import pandas as pd
import random
random.seed(49297)
from mimic3benchmark.util import *


def process_partition(args, partition, sample_rate=1.0, shortest_length=4.0, eps=1e-6):
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xty_triples = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for (patient_index, patient) in enumerate(patients):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                # empty label file
                if label_df.shape[0] == 0:
                    print("\n\t(empty label file)", patient, ts_filename)
                    continue

                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                # ts_lines = tsfile.readlines()
                # header = ts_lines[0]
                # ts_lines = ts_lines[1:]
                # event_times = [float(line.split(',')[0]) for line in ts_lines]

                # ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                #             if -eps < t < los + eps]

                ts = dataframe_from_csv(os.path.join(patient_folder, ts_filename), index_col=None)
                header = tsfile.readline()
                ts_lines = []
                for row in ts.itertuples():
                    lst = [row.Hours]
                    ts_lines.append(lst)
                event_times = [line[0] for line in ts_lines]

                ts_lines = ts.loc[(ts['Hours'] < los + eps) & (ts['Hours']>-eps)]



                event_times = [t for t in event_times
                               if -eps < t < los + eps]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                sample_times = np.arange(0.0, los + eps, sample_rate)

                sample_times = list(filter(lambda x: x > shortest_length, sample_times))

                # At least one measurement
                sample_times = list(filter(lambda x: x > event_times[0], sample_times))

                output_ts_filename = patient + "_" + ts_filename
                ts_lines.to_csv(os.path.join(output_dir, output_ts_filename), index = False)

                # with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                #     outfile.write(header)
                #     for line in ts_lines:
                #         outfile.write(line)

                for t in sample_times:
                    xty_triples.append((output_ts_filename, t, los - t))

        if (patient_index + 1) % 100 == 0:
            print("processed {} / {} patients".format(patient_index + 1, len(patients)), end='\r')

    print(len(xty_triples))
    if partition == "train":
        random.shuffle(xty_triples)
    if partition == "test":
        xty_triples = sorted(xty_triples)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,period_length,y_true\n')
        for (x, t, y) in xty_triples:
            listfile.write('{},{:.6f},{:.6f}\n'.format(x, t, y))


def main():
    parser = argparse.ArgumentParser(description="Create data for length of stay prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "test")
    process_partition(args, "train")


if __name__ == '__main__':
    main()