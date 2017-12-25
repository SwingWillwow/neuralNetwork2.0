import numpy as np


class CrossEntropyFunction(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1.0-y)*np.log(1.0-a)))

    @staticmethod
    def output_delta(z, a, y):
        return a-y
