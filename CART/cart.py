import numpy as np


def load_data_set(filename):
    data_mat = []
    input_file = open(filename)
    for line in input_file.readlines():
        curr_line = line.strip().split('\t')
        float_line = list(map(float, curr_line))
        data_mat.append(float_line)
    return data_mat


def bi_split_data_set(data_set, feature, value):
    mat0 = data_set[np.nonzero(data_set[:, feature] > value)[0], :]
    mat1 = data_set[np.nonzero(data_set[:, feature] <= value)[0], :]
    return mat0, mat1


def reg_leaf(data_set):
    return np.mean(data_set[:, -1])


def reg_err(data_set):
    return np.var(data_set[:, -1]) * np.shape(data_set)[0]


def reg_tree_eval(model, _):
    return float(model)


def model_leaf(data_set):
    ws, x, y = lin_solve(data_set)
    return ws


def model_err(data_set):
    ws, x, y = lin_solve(data_set)
    y_hat = x * ws
    return np.sum(np.power(y - y_hat, 2))


def model_tree_eval(model, data):
    n = np.shape(data)[1]
    x = np.mat(np.ones((1, n+1)))
    x[:, 1:n+1] = data
    return float(x * model)


def lin_solve(data_set):
    m, n = np.shape(data_set)
    x = np.mat(np.ones((m, n)))
    y = np.mat(np.ones((m, 1)))

    x[:, 1:n] = data_set[:, 0:n-1]
    y = data_set[:, -1]

    x_squared = x.T * x
    if np.linalg.det(x_squared) == 0.0:
        raise NameError("Singular matrix, cannot do inverse.")
    ws = x_squared.I * (x.T * y)
    return ws, x, y


def create_tree(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    feat, val = choose_best_split(data_set, leaf_type, err_type, ops)
    if feat is None:
        return val

    ret_tree = {'sp_idx': feat, 'sp_val': val}
    l_set, r_set = bi_split_data_set(data_set, feat, val)
    ret_tree['left'] = create_tree(l_set, leaf_type, err_type, ops)
    ret_tree['right'] = create_tree(r_set, leaf_type, err_type, ops)
    
    return ret_tree


def choose_best_split(data_set,
                      leaf_type=reg_leaf,
                      err_type=reg_err,
                      ops=(1, 4)):
    
    deduct_tolerance = ops[0]
    min_ct = ops[1]

    # all labels are of the same value, no need to split and return
    if len(set(data_set[:, -1].T.tolist()[0])) == 1:
        return None, leaf_type(data_set)

    m, n = np.shape(data_set)
    err = err_type(data_set)
    min_err = np.inf
    min_idx = 0
    min_value = 0

    for i in range(n - 1):  # every feature
        for split_val in set(data_set[:, i].T.tolist()[0]):  # every value
            mat0, mat1 = bi_split_data_set(data_set, i, split_val)
            if (np.shape(mat0)[0] < min_ct) or (np.shape(mat1)[0] < min_ct):
                continue
            curr_err = err_type(mat0) + err_type(mat1)
            if curr_err < min_err:
                min_idx = i
                min_value = split_val
                min_err = curr_err

    if (err - min_err) < deduct_tolerance:
        return None, leaf_type(data_set)  # not worth split
    mat0, mat1 = bi_split_data_set(data_set, min_idx, min_value)
    if (np.shape(mat0)[0] < min_ct) or (np.shape(mat1)[0] < min_ct):
        return None, leaf_type(data_set)
    return min_idx, min_value


def is_tree(obj):
    return type(obj).__name__ == 'dict'


def get_mean(tree):
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    return (tree['left'] + tree['right']) / 2


def prune(tree, test_data):
    if np.shape(test_data)[0] == 0:
        return get_mean(tree)

    if is_tree(tree['right']) or is_tree(tree['left']):
        l_set, r_set = bi_split_data_set(test_data,
                                         tree['sp_idx'],
                                         tree['sp_val'])
        if is_tree(tree['left']):
            tree['left'] = prune(tree['left'], l_set)
        if is_tree(tree['right']):
            tree['right'] = prune(tree['right'], r_set)

    # This logic is convoluted.
    if not is_tree(tree['left']) and not is_tree(tree['right']):
        l_set, r_set = bi_split_data_set(test_data,
                                         tree['sp_idx'],
                                         tree['sp_val'])
        err_no_merge = (np.sum(np.power(l_set[:, -1] - tree['left'], 2)) +
                        np.sum(np.power(r_set[:, -1] - tree['right'], 2)))
        tree_mean = (tree['left'] + tree['right']) / 2.0
        err_merge = np.sum(np.power(test_data[:, -1] - tree_mean, 2))
        if err_merge < err_no_merge:
            print("merging")
            return tree_mean
        else:
            return tree
    else:
        return tree


def tree_forecast(tree, data, model_eval=reg_tree_eval):
    if not is_tree(tree):
        return model_eval(tree, data)
    if data[tree['sp_idx']] > tree['sp_val']:
        if is_tree(tree['left']):
            return tree_forecast(tree['left'], data, model_eval)
        else:
            return model_eval(tree['left'], data)
    else:
        if is_tree(tree['right']):
            return tree_forecast(tree['right'], data, model_eval)
        else:
            return model_eval(tree['right'], data)


def create_forecast(tree, test_data, model_eval=reg_tree_eval):
    m = len(test_data)
    y_hat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        y_hat[i, 0] = tree_forecast(tree, np.mat(test_data[i]), model_eval)
    return y_hat

