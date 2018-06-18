"""
Logistic regression module.
"""

import numpy as np
import matplotlib.pyplot as plt


def load_data_set():
    data_mat = []
    label_mat = []

    data_file = open('testSet.txt')
    for line in data_file.readlines():
        line_ar = line.strip().split()
        data_mat.append([1.0, float(line_ar[0]), float(line_ar[1])])
        label_mat.append(int(line_ar[2]))
    return data_mat, label_mat


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def grad_ascent(input_mat, labels):
    data_mat = np.mat(input_mat)
    label_mat = np.mat(labels).transpose()

    m, n = np.shape(data_mat)
    alpha = 0.001
    max_cycles = 500
    weights = np.ones((n, 1))

    for k in range(max_cycles):
        h = sigmoid(data_mat * weights)
        error = label_mat - h
        weights += alpha * data_mat.transpose() * error  # calc gradient ascent

    return weights


def plot_best_fit(weights):
    data_mat, label_mat = load_data_set()
    data_arr = np.array(data_mat)
    n = np.shape(data_arr)[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []

    for i in range(n):
        if int(label_mat[i]) == 1:
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1]*x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stoc_grad_ascent(data_mat, labels):
    m, n = np.shape(data_mat)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(data_mat[i] * weights))
        error = labels[i] - h
        weights += alpha * error * data_mat[i]

    return weights


def stoc_grad_ascent_improved(data_mat, labels, num_iter=150):
    m, n = np.shape(data_mat)
    weights = np.ones(n)
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0+j+i) + 0.0001
            rand_index = int(np.random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_mat[rand_index] * weights))
            error = labels[rand_index] - h
            weights += alpha * error * data_mat[rand_index]
            del data_index[rand_index]

    return weights


def classify(data, weights):
    y = sigmoid(sum(data * weights))
    if y > 0.5:
        return 1.0
    return 0.0


def colic_test():
    training_file = open('horseColicTraining.txt')
    test_file = open('horseColicTest.txt')
    training_set = []
    labels = []

    for line in training_file.readlines():
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        training_set.append(line_arr)
        labels.append(float(curr_line[21]))

    train_weights = stoc_grad_ascent_improved(np.array(training_set),
                                              labels,
                                              200)
    error_count = 0
    num_test_vec = 0.0
    for line in test_file.readlines():
        num_test_vec += 1.0
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))

        result = classify(np.array(line_arr), train_weights)
        if int(result) != int(curr_line[21]):
            error_count += 1

    error_rate = float(error_count) / num_test_vec
    print("error rate is %f" % error_rate)
    return error_rate


def multi_test():
    num_tests = 10
    error_sum = 0.0
    for k in range(num_tests):
        error_sum += colic_test()
    print("after %d iterations, average error rate is %f" %
          (num_tests, error_sum / float(num_tests)))
