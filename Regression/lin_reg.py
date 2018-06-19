import numpy as np


def load_data_set(filename):
    """ Read data set from file system. """

    input_file = open(filename)

    num_features = len(input_file.readline().split('\t')) - 1
    input_file.seek(0)
    data_mat = []
    label_mat = []

    for line in input_file.readlines():
        line_arr = []
        curr_line = line.strip().split('\t')
        for i in range(num_features):
            line_arr.append(float(curr_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(curr_line[-1]))

    return data_mat, label_mat


def standard_reg(x_arr, y_arr):
    """Standard linear regression.
    Returns a weight vector.
    """

    x = np.mat(x_arr)
    y = np.mat(y_arr).T

    product = x.T * x
    if np.linalg.det(product) == 0.0:
        print("Matrix is singular, cannot do inverse.")
        return

    return product.I * (x.T * y)
