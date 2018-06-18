"""
Decision tree module.
"""

import operator
import math


def create_data_set():
    """Generate a simple test data set."""

    data_set = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def shannon_entropy(data_set):
    """Calculate Shannon entropy."""

    num_entries = len(data_set)
    label_dict = {}

    for feature_vec in data_set:
        curr_label = feature_vec[-1]
        if curr_label not in label_dict:
            label_dict[curr_label] = 0
        label_dict[curr_label] += 1

    entropy = 0.0
    for key in label_dict:
        prob = float(label_dict[key]) / num_entries
        entropy -= prob * math.log(prob, 2)

    return entropy


def split_data_set(data_set, axis, value):
    """Split set according to specified feature and value."""

    result = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            result.append(feat_vec[:axis] + feat_vec[axis+1:])

    return result


def best_feat_to_split(data_set):
    """Iterate all features and find out the best one to split on."""

    num_features = len(data_set[0]) - 1
    base_entropy = shannon_entropy(data_set)
    best_info_gain = 0.0
    best_feature = -1

    for i in range(num_features):
        feat_vals = [example[i] for example in data_set]
        uniq_feat_vals = set(feat_vals)
        new_entropy = 0.0

        for val in uniq_feat_vals:
            sub_data_set = split_data_set(data_set, i, val)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * shannon_entropy(sub_data_set)

        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i

    return best_feature


def majority_count(class_list):
    """Return feature of majority node."""

    class_count = {}
    for vote in class_list:
        if vote not in class_count:
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1),
                                reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    """Main method to generate decision tree."""

    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_count(class_list)

    best_feature = best_feat_to_split(data_set)
    best_feature_label = labels[best_feature]

    tree = {best_feature_label: {}}
    del labels[best_feature]
    feat_values = [example[best_feature] for example in data_set]
    uniq_values = set(feat_values)

    for val in uniq_values:
        sub_labels = labels[:]
        sub_ds = split_data_set(data_set, best_feature, val)
        tree[best_feature_label][val] = create_tree(sub_ds, sub_labels)
    return tree


def classify(input_tree, feat_labels, test_vec):
    """Run classification on input tree."""

    first_str = list(input_tree)[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    key = test_vec[feat_index]
    value_of_feat = second_dict[key]
    if isinstance(value_of_feat, dict):
        return classify(value_of_feat, feat_labels, test_vec)
    return value_of_feat


def store_tree(input_tree, filename):
    """Dump tree to file."""

    import pickle
    file = open(filename, 'wb')
    pickle.dump(input_tree, file)
    file.close()


def grab_tree(filename):
    """Read tree from file."""

    import pickle
    file = open(filename, 'rb')
    return pickle.load(file)
