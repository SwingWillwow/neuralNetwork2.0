import numpy as np
from network_components.activate_functions.sigmoid_function import Sigmoid


class QuadraticFunction(object):
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def output_delta(z, a, y):
        return (a-y) * Sigmoid.prime(z)
