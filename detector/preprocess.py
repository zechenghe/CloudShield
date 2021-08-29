'''
Covert csv data to npy arrays of shape [TimeFrame, Features].
'''

import argparse
import numpy as np
import os

import utils


def remove_outlier(data, window_size):
    mu = np.mean(data, axis=0)
    std = np.mean(data, axis=0)

    th = 3 * std
    data = (np.abs(data-mu) > th) * mu + (np.abs(data-mu) <= th) * data


    kernel = np.ones(window_size) / np.float32(window_size)
    data = [np.convolve(data[:, i], kernel, mode='same') for i in range(data.shape[-1])]
    data = np.array(data).T

    return data

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str, default = "../perf/data/core0/100us/", help='The directory of data')
parser.add_argument('--window_size', type = int, default = 500, help='Window size for moving average')
parser.add_argument('--file_name', type = str, default = None, help='The directory of data')
args = parser.parse_args()

data_dir = args.data_dir
file_name = args.file_name
if file_name != None:
    data = utils.read_csv_file(data_dir+file_name, dtype=np.float128)
    data = remove_outlier(data, args.window_size)
    np.save(data_dir + "".join(file_name.split('.')[:-1]) + '.npy', data)
else:
    for f in os.listdir(data_dir):
        extension = f.split('.')[-1]
        if extension == 'csv':
            data = utils.read_csv_file(data_dir+f, dtype=np.float128)
            #print(data)
            time_stamp = np.expand_dims(data[:, -1], axis=1)
            data = remove_outlier(data[:, :-1], args.window_size)
            data = np.concatenate((data, time_stamp), axis=-1)[args.window_size : -args.window_size]
            #n_ins_average = np.mean(data[:, 0])
            #data = data / n_ins_average
            np.save(data_dir + "".join(f.split('.')[:-1]) + '.npy', data)
