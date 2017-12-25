"""
dataLoader
~~~~~~~~~
this module used to load data form MNIST image data.
"""
#  standard libraries
import gzip
import pickle
import time
#  third-party libraries
import numpy as np


def load_original_data():
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="bytes")  # set encoding to bytes
    f.close()
    return training_data, validation_data, test_data


"""
this  math change tuple data into numpy ndarray
"""


def load_data():
    tr_data, vl_data, ts_data = load_original_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_data[0]]
    training_results = [get_vertical_result(y) for y in tr_data[1]]
    training_data = list(zip(training_inputs,training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in vl_data[0]]
    validation_results = vl_data[1]
    validation_data = list(zip(validation_inputs, validation_results))
    test_inputs = [np.reshape(x, (784, 1)) for x in ts_data[0]]
    test_results = ts_data[1]
    test_data = list(zip(test_inputs, test_results))
    print('load data use: %.3f seconds' % time.process_time())
    return training_data, validation_data, test_data


def get_vertical_result(ret):
    vertical_ret = np.zeros((10, 1))
    vertical_ret[ret] = 1.0
    return vertical_ret

