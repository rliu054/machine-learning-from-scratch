"""
AdaBoost module.
"""

import numpy as np
import matplotlib.pyplot as plt


def load_simple_data():
    """ Dummy data. """

    dat_mat = np.matrix([[1., 2.1],
                         [2., 1.1],
                         [1.3, 1.],
                         [1., 1.],
                         [2., 1.]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dat_mat, class_labels


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


def stump_classify(data_mat, dim, thresh_val, thresh_ineq):
    """ Classification using decision stump. Given dimension to cut,
    threshold, and which side to take.
    """

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
        Best estimate after training, (m, 1) dimension array.

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
                # print("dim %d, thresh %.2f, thresh ineqal: %s, the weighted "
                #       "error is %.3f" % (i, thresh_vals, ineq, weighted_err))

                if weighted_err < min_err:
                    min_err = weighted_err
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_vals
                    best_stump['ineq'] = ineq

    return best_stump, min_err, best_class_est


def ada_boost_decision_stump(data_arr, class_labels, num_iter=40):
    """Boosting on a weak algorithm, namely, decision stump.
    You can change it to some other algorithms.

    ---
    returns:
        An array of weak classifiers along with their alpha value, threshold,
        inequality operator, dimension info. This will be required to classify
        input data.
    """

    weak_class_arr = []
    m = np.shape(data_arr)[0]
    example_weights = np.mat(np.ones((m, 1)) / m)  # equal weights
    agg_class_est = np.mat(np.zeros((m, 1)))

    for i in range(num_iter):
        best_stump, error, class_est = build_stump(data_arr,
                                                   class_labels,
                                                   example_weights)
        # print("example weights: ", example_weights.T)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)
        # print("class est: ", class_est.T)

        # scale down correct examples and scale down incorrect ones
        exponent = np.multiply(-1 * alpha * np.mat(class_labels).T, class_est)
        example_weights = np.multiply(example_weights,
                                      np.exp(exponent)) / example_weights.sum()
        agg_class_est += alpha * class_est
        # print("agg class est: ", agg_class_est.T)
        agg_errors = np.multiply(np.sign(agg_class_est) !=
                                 np.mat(class_labels).T, np.ones((m, 1)))
        error_rate = agg_errors.sum() / m
        # print("total error: ", error_rate)
        if error_rate == 0.0:
            break

    return weak_class_arr, agg_class_est


def ada_classify(data, classifier_arr):
    """Classification using AdaBoost.

    data:
        Data instances to be classified.

    classifier_arr:
        An array of weak classifiers.
    """

    data_mat = np.mat(data)
    m = np.shape(data_mat)[0]
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(len(classifier_arr)):
        # run the data through multiple weak classifiers
        class_est = stump_classify(data_mat, classifier_arr[i]['dim'],
                                   classifier_arr[i]['thresh'],
                                   classifier_arr[i]['ineq'])
        agg_class_est += classifier_arr[i]['alpha'] * class_est
        # print(agg_class_est)

    return np.sign(agg_class_est)


def plot_roc(pred_strengths, class_labels):
    cur = (1.0, 1.0)
    y_sum = 0.0
    num_pos_class = np.sum(np.array(class_labels) == 1.0)
    y_step = 1 / float(num_pos_class)
    x_step = 1 / float(len(class_labels) - num_pos_class)
    sorted_indices = pred_strengths.argsort()

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sorted_indices.tolist()[0]:
        if class_labels[index] == 1.0:
            del_x = 0
            del_y = y_step
        else:
            del_x = x_step
            del_y = 0
            y_sum += cur[1]

        ax.plot([cur[0], cur[0] - del_x], [cur[1], cur[1] - del_y], c='b')
        cur = (cur[0] - del_x, cur[1] - del_y)

    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", y_sum * x_step)