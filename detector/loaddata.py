import torch
import numpy as np
import scipy.signal

from utils import read_npy_data_single_flle

def load_data_split(split = (0.4, 0.2, 0.4), data_dir = "data/", file_name = 'baseline.npy'):

    assert (len(split) == 3) and (sum(split) == 1.0), "Data split error..."

    normal_data_path = data_dir + file_name
    data = read_npy_data_single_flle(normal_data_path)

    assert len(data.shape) == 2, "Data should be in shape (TimeFrame, Features)"

    total_length = data.shape[0]
    training_length = int(total_length * split[0])
    ref_length = int(total_length * split[1])
    testing_length = int(total_length * split[2])

    training_normal = data[: training_length, :]
    ref_normal = data[training_length: training_length + ref_length, :]
    testing_normal = data[training_length + ref_length :, :]

    return np.float32(training_normal), np.float32(ref_normal), np.float32(testing_normal)


def load_data_all(data_dir, file_name):

    abnormal_data_path = data_dir + file_name
    data = read_npy_data_single_flle(abnormal_data_path)

    assert len(data.shape) == 2, "Data should be in shape (TimeFrame, Features)"

    return np.float32(data)

def load_normal_dummydata():

    # Assume normal sequence is square wave with noise. Try your own data

    T = 10000
    t = np.linspace(0, 1000, num = T)
    training_data = np.float32(np.expand_dims(scipy.signal.square(2 * np.pi * t), axis = 1))
    val_data = np.float32(np.expand_dims(scipy.signal.square(2 * np.pi * t), axis = 1))
    testing_data = np.float32(np.expand_dims(scipy.signal.square(2 * np.pi * t), axis = 1))

    training_data += np.random.normal(loc=0.0, scale=0.2, size = (T, 1))
    val_data += np.random.normal(loc=0.0, scale=0.2, size = (T, 1))
    testing_data += np.random.normal(loc=0.0, scale=0.2, size = (T, 1))

    return training_data, val_data, testing_data


def load_abnormal_dummydata():

    T = 10000
    t = np.linspace(0, 500, num = T)
    #t = np.linspace(0, 1, T, endpoint=False)
    #training_data = scipy.signal.square(2 * np.pi * 2 * t)
    abnormal_data = np.float32(np.expand_dims(scipy.signal.square(2 * np.pi * t), axis = 1))
    abnormal_data += np.float32(np.random.normal(loc=0.0, scale=0.2, size = (T, 1)))

    return abnormal_data
