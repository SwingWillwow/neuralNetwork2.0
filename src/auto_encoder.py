from network_components.data_loader import load_data
from network_components.network import Network
from network_components.cost_functions.quadratic_function import QuadraticFunction
import numpy as np
import datetime
import json
import os
import sys
import matplotlib.pyplot as plt

def load(file_name):
    f = open(file_name, "r")
    data = json.load(f)
    f.close()
    # get cost_function name
    cost_name = data["cost"]
    # in cost_functions_path search such cost function
    cost_fuctions_path = "./network_components/cost_functions/"
    # get all the dir in the path
    parents = os.listdir(cost_fuctions_path)
    cost_func = None
    # get cost_function
    for parent in parents:
        child = os.path.join(cost_fuctions_path, parent)
        child = str(child)
        if child.endswith(".py"):
            child = child.replace("/", ".")
            child = child[2:-3]
            try:
                cost_func = getattr(sys.modules[child], cost_name)
                break
            except AttributeError:
                continue
    # load the network
    network = Network(data["layer"], cost_func)
    network.weights = [np.array(w) for w in data["weights"]]
    network.biases = [np.array(b) for b in data["biases"]]
    return network


def get_data():
    td, vd, test_d = load_data()
    train_input, train_target = zip(*td)
    valid_input, valid_target = zip(*vd)
    train_d = list(zip(train_input, train_input))
    valid_d = list(zip(valid_input, valid_input))
    return train_d, valid_d


def get_Matrix_into_vectors(data):
    data_len = len(data[0])
    data_size = len(data)
    ret = np.zeros((data_len, data_size, 1))
    for i in range(data_len):
        ret[i] = np.array(data[:, i]).reshape((data_size, 1))
    return ret


def DAC_get_input_matrix(data):
    if len(data) > 2:
        n = len(data)
        n = n//2
        left_data = DAC_get_input_matrix(data[0:n])
        right_data = DAC_get_input_matrix(data[n:])
        return np.hstack((left_data, right_data))
    elif len(data) == 2:
        return np.hstack((data[0], data[1]))
    elif len(data) == 1:
        return data[0]


def get_accuracy_pic(accuracy):
    x = np.arange(1, len(accuracy)+1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    accuracy = np.array(accuracy)
    ax.plot(x, [y/100.00 for y in accuracy])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('accuracy on the evaluate data')
    f = open("../data/accuracy_pic_auto_encoder.jpg", "w")
    plt.savefig(f, format='jpg')
    plt.show()


def get_cost_pic(cost):
    x = np.arange(1, len(cost)+1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cost = np.array(cost)
    ax.plot(x, [y for y in cost])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the evaluate data')
    f = open("../data/cost_pic._auto_encoder.jpg", "w")
    plt.savefig(f, format='jpg')
    plt.show()


train_data, valid_data = get_data()
net = Network([784, 50, 784])
net.train(train_data, 100, 30, 0.25, 5, valid_data)
net.save("../data/"+str(datetime.date.today())+"_auto_encoder"+".json")


# net = load("../data/2017-12-28.json")
W = net.weights[0]
b = net.biases[0]
td, vd, test_d = load_data()
train_input, train_target = zip(*td)
evaluate_input, evaluate_target = zip(*vd)
train_input_reshaped = DAC_get_input_matrix(train_input)
evaluate_input_reshaped = DAC_get_input_matrix(evaluate_input)
low_dims_train_input = np.dot(W, train_input_reshaped) + b
low_dims_evaluate_input = np.dot(W, evaluate_input_reshaped) + b
low_dims_train_input = get_Matrix_into_vectors(low_dims_train_input)
low_dims_evaluate_input = get_Matrix_into_vectors(low_dims_evaluate_input)
train_data = list(zip(low_dims_train_input, train_target))
evaluate_data = list(zip(low_dims_evaluate_input, evaluate_target))
net2 = Network([50, 100, 10])
evaluate_cost, evaluate_accuracy, training_cost, training_accuracy = \
    net2.train(train_data, 200, 30, 0.1, 5, evaluate_data=evaluate_data, monitor_evaluate_cost=True,
               monitor_evaluate_accuracy=True)
net.save("../data/"+str(datetime.date.today())+"auto_encoder_trained"+".json")
get_accuracy_pic(evaluate_accuracy)
get_cost_pic(evaluate_cost)


