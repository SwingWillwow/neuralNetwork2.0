import json
import random
import time
import sys
# third-party libraries
import numpy as np
from network_components.cost_functions.cross_entropy_function import CrossEntropyFunction
from network_components.cost_functions.quadratic_function import QuadraticFunction
from network_components.activate_functions.sigmoid_function import Sigmoid
import timeit

class Network(object):
    def __init__(self, layer, cost_function=CrossEntropyFunction):
        self.layer_size = len(layer)
        self.layer = layer
        self.biases = [np.random.randn(x, 1) for x in layer[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(layer[:-1], layer[1:])]
        self.cost_func = cost_function

    def feed_forward(self, a):
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a)+b
            a = Sigmoid.func(z)
        return a

    def mini_batch_back_propagation(self, mini_batch, eta, lmbda):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        mini_batch_size = len(mini_batch)
        mini_batch = np.array(list(zip(*mini_batch)))
        # set x to (784,30)
        x = mini_batch[0][0]
        x = np.zeros((len(x), 0))
        for single_x in mini_batch[0]:
            x = np.hstack((x, single_x))
        # set y to (10,30)
        y = mini_batch[1][0]
        y = np.zeros((len(y), 0))
        for single_y in mini_batch[1]:
            y = np.hstack((y, single_y))
        # feed forward to get all activations and zs
        activations = [x]
        activation = x
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = Sigmoid.func(z)
            activations.append(activation)
        delta = self.cost_func.output_delta(zs[-1], activations[-1], y)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = np.sum(delta, axis=1)
        nabla_b[-1] = nabla_b[-1].reshape(nabla_b[-1].size, 1)
        for l in range(2, self.layer_size):
            delta = np.dot(self.weights[-l+1].transpose(), delta)*Sigmoid.prime(zs[-l])
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            nabla_b[-l] = np.sum(delta, axis=1)
            nabla_b[-l] = nabla_b[-l].reshape(nabla_b[-l].size, 1)
        return nabla_w, nabla_b

    def mini_batch_gradient_descend(self, mini_batch, eta, lmbda, training_data_len):
        nabla_w, nabla_b = self.mini_batch_back_propagation(mini_batch, eta, lmbda)
        self.weights = [w-(eta/len(mini_batch))*nw-(eta*(lmbda/training_data_len))*w
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def train(self, training_data, epochs, mini_batch_size, eta, lmbda=0.0,
              evaluate_data=None,
              monitor_evaluate_cost=False,
              monitor_evaluate_accuracy=False,
              monitor_training_cost=False,
              monitor_training_accuracy=False,
              monitor_error_rate=False,
              early_stopping_n=0):
        start_time = timeit.default_timer()
        best_accuracy = 0
        no_accuracy_change = 0
        evaluate_cost, evaluate_accuracy = [], []
        training_cost, training_accuracy = [], []
        training_data_len = len(training_data)
        evaluate_data_len = 0
        if evaluate_data:
            evaluate_data_len = len(evaluate_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, training_data_len, mini_batch_size)]
            for mini_batch in mini_batches:
                self.mini_batch_gradient_descend(mini_batch, eta, lmbda, training_data_len)
            print("Epoch{} complete.".format(i+1))
            if monitor_evaluate_cost:
                cost = self.total_cost(evaluate_data, lmbda, True)
                evaluate_cost.append(cost)
                print("cost in evaluate data: {:.3f}".format(cost))
            if monitor_evaluate_accuracy:
                accuracy = self.get_accuracy(evaluate_data)
                evaluate_accuracy.append(accuracy)
                print("accuracy in evaluate data: {} / {}".format(accuracy, evaluate_data_len))
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("cost in training data: {:.3f}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.get_accuracy(training_data, True)
                training_accuracy.append(accuracy)
                print("accuracy in training data: {} / {}".format(accuracy, training_data_len))
            if monitor_error_rate:
                pass
            if early_stopping_n > 0:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_accuracy_change = 0
                else:
                    no_accuracy_change += 1
                if no_accuracy_change == early_stopping_n:
                    print("Early stopping: no accuracy change in last {} epochs".format(early_stopping_n))
                    return evaluate_cost, evaluate_accuracy, training_cost, training_accuracy
        print()
        elasped = timeit.default_timer() - start_time
        print("train use " + change_time(elasped))
        return evaluate_cost, evaluate_accuracy, training_cost, training_accuracy

    def get_accuracy(self, data, convert=False):
        if convert:
            result = [(np.argmax(self.feed_forward(x)), np.argmax(y)) for x, y in data]
        else:
            result = [(np.argmax(self.feed_forward(x)), y) for x, y in data]
        return sum(int(x == y) for x, y in result)

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            if convert:
                y = get_vertical_form(y)
            a = self.feed_forward(x)
            cost += self.cost_func.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, file_name):
        data = {"layer": self.layer,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost_func.__name__)
        }
        f = open(file_name, "w")
        json.dump(data, f)
        f.close()


def get_vertical_form(y):
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e


def change_time(t):
    t = int(t)
    return '{} minutes {} seconds'.format(t // 60, t % 60)
