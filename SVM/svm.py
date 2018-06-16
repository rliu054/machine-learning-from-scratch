"""
SVM module.
"""

from time import sleep
import random
import numpy as np


def load_data_set(file_name):
    data_mat = []
    label_mat = []
    file = open(file_name)

    for line in file.readlines():
        line_arr = line.strip().split('\t')
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


def rand_j(index, m):
    rand = random.randrange(m)
    while (rand == index):
        rand = random.randrange(m)
    return rand


def clip_alpha(val, high, low):
    if val > high:
        return high
    if val < low:
        return low
    return val

