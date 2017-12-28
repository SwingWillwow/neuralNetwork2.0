from network_components.network import Network
import network_components.data_loader as loader
import time
import datetime
import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from network_components.cost_functions.quadratic_function import QuadraticFunction


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


def load_test():
    net = load("../data/2017-12-25.json")
    right = net.get_accuracy(test_data)
    print("accuracy in test_data. {} / {}".format(right, len(test_data)))
    cost = net.total_cost(test_data, 5, True)
    print("cost in test_data. {:.3f}".format(cost))


def get_accuracy_pic(accuracy):
    x = np.arange(1, len(accuracy)+1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    accuracy = np.array(accuracy)
    ax.plot(x, [y/100.00 for y in accuracy])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('accuracy on the evaluate data')
    f = open("../data/accuracy_pic_without_l2.jpg", "w")
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
    f = open("../data/cost_pic_without_l2.jpg", "w")
    plt.savefig(f, format='jpg')
    plt.show()


training_data, validation_data, test_data = loader.load_data()
net = Network([784, 100, 10],cost_function=QuadraticFunction)
epochs = input('please input the epochs.')
epochs = int(epochs)
mini_batch_size = int(input('please input the mini_batch_size.'))
eta = float(input('please input the learning rate.'))
lamda = float(input('please input the regularization parameter.'))
evaluate_cost, evaluate_accuracy, training_cost, training_accuracy = \
    net.train(training_data, epochs, mini_batch_size, eta, lamda, evaluate_data=validation_data,
              monitor_evaluate_cost=True, monitor_evaluate_accuracy=True)
net.save("../data/"+str(datetime.date.today())+"_origin_without_l2"+".json")
get_accuracy_pic(evaluate_accuracy)
get_cost_pic(evaluate_cost)


