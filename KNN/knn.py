"""
This module implements K nearest neighbour algorithm.
"""

from os import listdir
import operator
import numpy as np


def create_data_set():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(in_x, data_set, labels, k):
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_indicies = distances.argsort()

    class_count = {}
    for i in range(k):
        vote_ilabel = labels[sorted_dist_indicies[i]]
        class_count[vote_ilabel] = class_count.get(vote_ilabel, 0) + 1
        sorted_class_count = sorted(class_count.items(),
                                    key=operator.itemgetter(1),
                                    reverse=True)

    return sorted_class_count[0][0]


def file_to_matrix(filename):
    love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}

    data_file = open(filename)
    array_lines = data_file.readlines()
    num_of_lines = len(array_lines)
    return_mat = np.zeros((num_of_lines, 3))
    class_label_vector = []
    index = 0

    for line in array_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]
        if list_from_line[-1].isdigit():
            class_label_vector.append(int(list_from_line[-1]))
        else:
            class_label_vector.append(love_dictionary.get(list_from_line[-1]))
        index += 1

    return return_mat, class_label_vector


def img_to_vector(filename):
    vec = np.zeros((1, 1024))
    file = open(filename)
    for i in range(32):
        line = file.readline()
        for j in range(32):
            vec[0, 32*i + j] = int(line[j])
    return vec


def auto_norm(data_set):
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    norm_data_set = np.zeros(np.shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - np.tile(min_vals, (m, 1))
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))

    return norm_data_set, ranges, min_vals


def dating_class_test():
    ratio = 0.10
    dating_data_mat, dating_labels = file_to_matrix('data/datingTestSet.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify(norm_mat[i, :],
                                     norm_mat[num_test_vecs:m, :],
                                     dating_labels[num_test_vecs:m],
                                     3)
        print("the classifier came back with: %d, real answer is: %d"
              % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print("total error rate is: %f" % (error_count / float(num_test_vecs)))
    print(error_count)


def classify_person():
    return_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input("percentage of time playing games?"))
    ff_miles = float(input("frequent flier miles?"))
    ice_cream = float(input("liters of ice cream?"))
    dating_data_mat, dating_labels = file_to_matrix("data/datingTestSet2.txt")
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = np.array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify((in_arr - min_vals) / ranges,
                                 norm_mat,
                                 dating_labels, 3)

    print("You will like this person: ", return_list[classifier_result - 1])


# This is terrible Python code.
def handwriting_test():
    labels = []
    training_files = listdir('data/trainingDigits')
    m = len(training_files)
    training_mat = np.zeros((m, 1024))

    for i in range(len(training_files)):
        file_name = training_files[i]
        file_str = file_name.split('.')[0]
        class_num = int(file_str.split('_')[0])
        labels.append(class_num)
        training_mat[i, :] = img_to_vector('data/trainingDigits/%s'
                                           % file_name)

    test_files = listdir('data/testDigits')
    error_count = 0.0
    for i in range(len(test_files)):
        file_name = test_files[i]
        file_str = file_name.split('.')[0]
        class_num = int(file_str.split('_')[0])
        labels.append(class_num)
        test_vector = img_to_vector('data/trainingDigits/%s' % file_name)
        result = classify(test_vector, training_mat, labels, 3)

        print("classifier came back with %d, answer is %d"
              % (result, class_num))
        if result != class_num:
            error_count += 1.0
    print("total num of errors: %d" % error_count)
    print("total error rate is: %d" % (error_count / float(len(test_files))))
