import numpy as np


class Sigmoid(object):
    @staticmethod
    def func(z):
        return 1.0/(1.0+np.exp(-z))

    @staticmethod
    def prime(z):
        return Sigmoid.func(z)*(1.0-Sigmoid.func(z))
