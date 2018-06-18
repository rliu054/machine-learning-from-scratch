"""
AdaBoost module.
"""

import numpy as np


def load_simple_data():
    dat_mat = np.matrix([[1., 2.1],
                         [2., 1.1],
                         [1.3, 1.],
                         [1., 1.],
                         [2., 1.]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dat_mat, class_labels


def stump_classify(data_mat, dim, thresh_val, thresh_ineq):
    ret_arr = np.ones((np.shape(data_mat)[0], 1))
    if thresh_ineq == 'lt':
        ret_arr[data_mat[:, dim] <= thresh_val] = -1.0
    else:
        ret_arr[data_mat[:, dim] > thresh_val] = -1.0
    return ret_arr


def build_stump(data_arr, class_labels, example_weights):
    """A weak algorithm used for boosting.

    est_stump:
        Dictionary with keys including dimention and threshold of the stump.

    min_err:
        Current minimum error, scalar number.

    best_class_est:
        Best estimate after training, (m, 1) dimention array.

    example_weights:
        Weights for each training example, AdaBoost scales up incorrectly
        classified examples and scales down correctly classified ones.
        Usually initialized to equal weights.
    """

    data_mat = np.mat(data_arr)
    label_mat = np.mat(class_labels).T
    m, n = np.shape(data_mat)

    num_steps = 10.0
    best_stump = {}
    best_class_est = np.mat(np.zeros((m, 1)))
    min_err = np.inf

    for i in range(n):
        range_min = data_mat[:, i].min()
        range_max = data_mat[:, i].max()
        step_size = (range_max - range_min) / num_steps

        for j in range(-1, int(num_steps) + 1):
            for ineq in ['lt', 'gt']:
                thresh_vals = (range_min + float(j) * step_size)
                predicted_vals = stump_classify(data_mat, i, thresh_vals, ineq)
                err_arr = np.mat(np.ones((m, 1)))
                err_arr[predicted_vals == label_mat] = 0

                # Example weights make it way to error calculation.
                weighted_err = example_weights.T * err_arr
                print("dim %d, thresh %.2f, thresh ineqal: %s, the weighted"
                      "error is %.3f" % (i, thresh_vals, ineq, weighted_err))

                if weighted_err < min_err:
                    min_err = weighted_err
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thres'] = thresh_vals
                    best_stump['ineq'] = ineq

    return best_stump, min_err, best_class_est


def ada_boost_decision_stump(data_arr, class_labels, num_iter=40):
    """ Boosting on a weak algorithm -- decision stump. """

    weak_class_arr = []
    m = np.shape(data_arr)[0]
    example_weights = np.mat(np.ones((m, 1)) / m)  # equal weights
    agg_class_est = np.mat(np.zeros((m, 1)))

    for i in range(num_iter):
        best_stump, error, class_est = build_stump(data_arr,
                                                   class_labels,
                                                   example_weights)
        print("example weights: ", example_weights.T)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)
        print("class est: ", class_est.T)

        expon = np.multiply(-1 * alpha * np.mat(class_labels).T, class_est)
        example_weights = np.multiply(example_weights, np.exp(expon))
        example_weights = example_weights / example_weights.sum()
        agg_class_est += alpha * class_est
        print("agg class est: ", agg_class_est.T)
        agg_errors = np.multiply(np.sign(agg_class_est) !=
                                 np.mat(class_labels).T, np.ones((m, 1)))
        error_rate = agg_errors.sum() / m
        print("total error: ", error_rate)
        if error_rate == 0.0:
            break

    return weak_class_arr
