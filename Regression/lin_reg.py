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
    """Standard linear regression. Returns a weight vector."""

    x = np.mat(x_arr)
    y = np.mat(y_arr).T

    x_squared = x.T * x
    if np.linalg.det(x_squared) == 0.0:
        print("Matrix is singular, cannot do inverse.")
        return

    return x_squared.I * (x.T * y)


def local_weighted_lr(test_point, x_arr, y_arr, k=1.0):
    """Locally weighted linear regression.
    Gives more weights to locally adjacent examples.
    """

    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    m = np.shape(x_mat)[0]
    weights = np.mat(np.eye(m))

    for j in range(m):
        diff_mat = test_point - x_mat[j, :]
        weights[j, j] = np.exp(diff_mat * diff_mat.T / (-2.0 * (k ** 2)))

    x_squared = x_mat.T * (weights * x_mat)
    if np.linalg.det(x_squared) == 0.0:
        print("Matrix is singular, cannot do inverse.")
        return

    ws = x_squared.I * (x_mat.T * (weights * y_mat))
    return test_point * ws


def local_weight_lr_test(test_arr, x_arr, y_arr, k=1.0):
    m = np.shape(test_arr)[0]
    y_hat = np.zeros(m)
    for i in range(m):
        y_hat[i] = local_weighted_lr(test_arr[i], x_arr, y_arr, k)
    return y_hat


def squared_err(y_arr, y_hat_arr):
    return ((y_arr - y_hat_arr) ** 2).sum()


def ridge_reg(x_mat, y_mat, lam=0.2):
    x_squared = x_mat.T * x_mat
    denom = x_squared + np.eye(np.shape(x_mat)[1]) * lam

    if np.linalg.det(x_squared) == 0.0:
        print("Matrix is singular, cannot do inverse.")
        return
    ws = denom.I * (x_mat.T * y_mat)
    return ws


def ridge_test(x_arr, y_arr):

    # normalize data
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T

    y_mean = np.mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_means = np.mean(x_mat, 0)

    x_var = np.var(x_mat, 0)
    x_mat = (x_mat - x_means) / x_var

    num_test_pts = 30
    w_mat = np.zeros((num_test_pts, np.shape(x_mat)[1]))
    for i in range(num_test_pts):
        ws = ridge_reg(x_mat, y_mat, np.exp(i - 10))
        w_mat[i, :] = ws.T
    return w_mat


def regularize(x_mat):
    in_mat = x_mat.copy()
    in_means = np.mean(in_mat, 0)
    in_var = np.var(np.mat(in_mat), 0)
    in_mat = (in_mat - in_means) / in_var
    return in_mat


def stage_wise_reg(x_arr, y_arr, eps=0.01, num_iter=100):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    y_mean = np.mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_mat = regularize(x_mat)

    m, n = np.shape(x_mat)
    return_mat = np.zeros((num_iter, n))
    ws = np.zeros((n, 1))
    ws_max = ws.copy()

    for i in range(num_iter):
        print(ws.T)
        min_err = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                ws_test = ws.copy()
                ws_test[j] += eps * sign
                y_test = x_mat * ws_test
                err = squared_err(y_mat.A, y_test.A)
                if err < min_err:
                    min_err = err
                    ws_max = ws_test
        ws = ws_max.copy()
        return_mat[i, :] = ws.T
    return return_mat
