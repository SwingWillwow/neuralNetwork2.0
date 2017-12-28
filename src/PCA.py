import numpy as np
from network_components.data_loader import load_data
from network_components.network import Network
import matplotlib.pyplot as plt
import datetime
import timeit

def change_time(t):
    t = int(t)
    return '{} minutes {} seconds'.format(t // 60, t % 60)

# data: each col contains a sample
def centralized(data):
    mean_value = np.mean(data, axis=1)
    mean_value = mean_value.reshape((mean_value.shape[0], 1))
    centralized_data = data - mean_value
    return centralized_data, mean_value
    # normalization
    # var_value = np.var(data, axis=0)
    # tmp_data = np.nan_to_num(centralized_data / var_value)
    # return tmp_data, mean_value


def pca_n(data, n):
    start_time = timeit.default_timer()
    cov_var = np.dot(data, data.T)
    eig_values, eig_vectors = np.linalg.eig(cov_var)
    eig_value_indices = np.argsort(eig_values)
    max_n_eig_value_indices = eig_value_indices[-1:-(n + 1):-1]
    max_n_eig_vec = eig_vectors[:, max_n_eig_value_indices]
    elapsed = timeit.default_timer()-start_time
    print("PCA used "+change_time(elapsed))
    return max_n_eig_vec


def pca_percent(data, percent=0.95):
    data = np.array(data)
    data, mean = centralized(data)
    cov_var = np.dot(data, data.T)
    eig_values, eig_vectors = np.linalg.eig(cov_var)
    eig_value_indices = np.argsort(eig_values)
    sum_eig_value = float(np.sum(eig_values, axis=0))
    tmp_sum_eig_value = 0.0
    n = 0
    max_order_eig_value_indices = eig_value_indices[::-1]
    for eig_value in eig_values[max_order_eig_value_indices]:
        tmp_sum_eig_value += eig_value
        n += 1
        if tmp_sum_eig_value / sum_eig_value > percent:
            break
    max_n_eig_value_indices = max_order_eig_value_indices[0:n:1]
    max_n_eig_vec = eig_vectors[:, max_n_eig_value_indices]
    low_dim_data = np.dot(max_n_eig_vec.T, data)
    # reconstruct_data = np.dot(low_dim_data, max_n_eig_vec.T) + mean
    return low_dim_data, max_n_eig_vec, mean  # , reconstruct_data


def DAC_get_input_matrix(data):
    if len(data) > 2:
        n = len(data)
        n = n // 2
        left_data = DAC_get_input_matrix(data[0:n])
        right_data = DAC_get_input_matrix(data[n:])
        return np.hstack((left_data, right_data))
    elif len(data) == 2:
        return np.hstack((data[0], data[1]))
    elif len(data) == 1:
        return data[0]


def get_Matrix_into_vectors(data):
    data_len = len(data[0])
    data_size = len(data)
    ret = np.zeros((data_len, data_size, 1))
    for i in range(data_len):
        ret[i] = np.array(data[:, i]).reshape((data_size, 1))
    return ret


def get_accuracy_pic(accuracy):
    x = np.arange(1, len(accuracy) + 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    accuracy = np.array(accuracy)
    ax.plot(x, [y / 100.00 for y in accuracy])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('accuracy on the evaluate data')
    f = open("../data/accuracy_pic_PCA.jpg", "w")
    plt.savefig(f, format='jpg')
    plt.show()


def get_cost_pic(cost):
    x = np.arange(1, len(cost) + 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cost = np.array(cost)
    ax.plot(x, [y for y in cost])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the evaluate data')
    f = open("../data/cost_pic._PCA.jpg", "w")
    plt.savefig(f, format='jpg')
    plt.show()


test_data = [
    [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1],
    [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]
]

# test_data = np.array(test_data)
# centralized_data, mean = centralized(test_data)
# W = pca_n(centralized_data, 1)
# low_dim_data = np.dot(W.T, centralized_data)
# print(low_dim_data)


train_data, validation_data, t_data = load_data()
train_input, train_target = zip(*train_data)
evaluate_input, evaluate_target = zip(*validation_data)
train_input_reshape = DAC_get_input_matrix(train_input)
evaluate_input_reshape = DAC_get_input_matrix(evaluate_input)
train_input_reshape, mean = centralized(train_input_reshape)
W_1 = pca_n(train_input_reshape, 50)
low_dim_data = np.dot(W_1.T, train_input_reshape)
low_dim_data = np.array(low_dim_data)
evaluate_input_reshape = evaluate_input_reshape - mean
low_dim_evaluate_data = np.dot(W_1.T, evaluate_input_reshape)
low_dim_input = get_Matrix_into_vectors(low_dim_data)
low_dim_evaluate_input = get_Matrix_into_vectors(low_dim_evaluate_data)
train_data = list(zip(low_dim_input, train_target))
evaluate_data = list(zip(low_dim_evaluate_input, evaluate_target))
net = Network([50, 100, 10])
evaluate_cost, evaluate_accuracy, training_cost, training_accuracy = \
    net.train(train_data, 200, 30, 0.1, 5, evaluate_data=evaluate_data, monitor_evaluate_cost=True,
              monitor_evaluate_accuracy=True)
net.save("../data/" + str(datetime.date.today()) + "PCA" + ".json")
get_accuracy_pic(evaluate_accuracy)
get_cost_pic(evaluate_cost)
