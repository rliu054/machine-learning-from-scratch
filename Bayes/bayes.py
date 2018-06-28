"""
This module implements a Naive Bayes classifier.
"""

import operator
import re

import numpy as np


def load_data_set():
    posting_list = [[
        'my', 'dog', 'has', 'flea', 'problems', 'help', 'please'
    ], ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'], [
        'my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'
    ], ['stop', 'posting', 'stupid', 'worthless', 'garbage'], [
        'mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'
    ], ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    labels = [0, 1, 0, 1, 0, 1]
    return posting_list, labels


def create_vocab_list(data_set):
    vocab_set = set([])
    for doc in data_set:
        vocab_set |= set(doc)
    return list(vocab_set)


def set_of_words2vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print("word %s not in my vocab!" % word)
    return return_vec


def bag_of_words2vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


def train(train_mat, train_category):
    num_train_docs = len(train_mat)
    num_words = len(train_mat[0])

    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0

    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_mat[i]
            p1_denom += sum(train_mat[i])
        else:
            p0_num += train_mat[i]
            p0_denom += sum(train_mat[i])

    p0_vec = np.log(p0_num / p0_denom)
    p1_vec = np.log(p1_num / p1_denom)

    return p0_vec, p1_vec, p_abusive


def classify(input_vec, p0_vec, p1_vec, p_class1):
    p0 = sum(input_vec * p0_vec) + np.log(1.0 - p_class1)
    p1 = sum(input_vec * p1_vec) + np.log(p_class1)
    if p1 > p0:
        return 1
    return 0


def testing():
    post_list, class_list = load_data_set()
    vocab_list = create_vocab_list(post_list)
    train_mat = []
    for post in post_list:
        train_mat.append(set_of_words2vec(vocab_list, post))

    p0_vec, p1_vec, p_abusive = train(
        np.array(train_mat), np.array(class_list))

    test_entry = ['love', 'stupid', 'stupid']
    this_doc = np.array(set_of_words2vec(vocab_list, test_entry))
    print(test_entry, 'classified as: ',
          classify(this_doc, p0_vec, p1_vec, p_abusive))
    test_entry = ['stupid', 'love']
    this_doc = np.array(set_of_words2vec(vocab_list, test_entry))
    print(test_entry, 'classified as: ',
          classify(this_doc, p0_vec, p1_vec, p_abusive))


def parse_text(string):
    token_list = re.split(r'\W*', string)
    return [token.lower() for token in token_list if len(token) > 2]


def spam_test():
    doc_list = []
    class_list = []
    full_text = []

    for i in range(1, 26):
        ham_file = open('email/ham/%d.txt' % i, encoding="ISO-8859-1")
        word_list = parse_text(ham_file.read())
        doc_list.append(word_list)
        full_text.append(word_list)
        class_list.append(0)

        spam_file = open('email/spam/%d.txt' % i, encoding="ISO-8859-1")
        word_list = parse_text(spam_file.read())
        doc_list.append(word_list)
        full_text.append(word_list)
        class_list.append(1)

    vocab_list = create_vocab_list(doc_list)
    training_set = range(50)
    test_set = []

    # reserve 20% for validation purpose
    for i in range(10):
        rand_index = int(np.random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del list(training_set)[rand_index]

    train_mat = []
    train_class = []
    for doc_index in training_set:
        train_mat.append(bag_of_words2vec(vocab_list, doc_list[doc_index]))
        train_class.append(class_list[doc_index])

    p0_vec, p1_vec, p_spam = train(np.array(train_mat), np.array(train_class))

    error_count = 0
    for doc_index in test_set:
        word_vec = bag_of_words2vec(vocab_list, doc_list[doc_index])
        result = classify(np.array(word_vec), p0_vec, p1_vec, p_spam)
        if result != class_list[doc_index]:
            error_count += 1
            print("classification error", doc_list[doc_index])

    print("the error rate is: ", float(error_count) / len(test_set))


def calc_most_freq(vocab_list, full_text):
    freq_dic = {}
    for token in vocab_list:
        freq_dic[token] = full_text.count(token)
    sorted_freq = sorted(
        freq_dic.items(), key=operator.itemgetter(1), reverse=True)
    # return sorted_freq[:30]
    return sorted_freq[:30]


def local_words(feed_a, feed_b):
    doc_list = []
    class_list = []
    full_text = []

    min_len = min(len(feed_a['entries']), len(feed_b['entries']))
    print(len(feed_a['entries']), len(feed_b['entries']))
    for i in range(min_len):
        word_list = parse_text(feed_a['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append('feed_a')

        word_list = parse_text(feed_b['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append('feed_b')

    vocab_list = create_vocab_list(doc_list)
    top_30_words = calc_most_freq(vocab_list, full_text)
    for word in top_30_words:
        if word[0] in vocab_list:
            vocab_list.remove(word[0])

    training_set = range(2 * min_len)

    test_set = []
    for i in range(20):
        rand_index = int(np.random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del list(training_set)[rand_index]

    train_mat = []
    train_classes = []
    for doc_index in training_set:
        train_mat.append(bag_of_words2vec(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    vec_a, vec_b, p_b = train(np.array(train_mat), np.array(train_classes))

    error_count = 0
    for doc_index in test_set:
        word_vector = bag_of_words2vec(vocab_list, doc_list[doc_index])
        result = classify(np.array(word_vector), vec_a, vec_b, p_b)
        if result != class_list[doc_index]:
            error_count += 1

    print('the error count is: ', float(error_count))
    print('the error rate is: ', float(error_count) / len(test_set))
    return vocab_list, vec_a, vec_b


def get_top_words(sf, seattle):
    vocab_list, p0_vec, p1_vec = local_words(sf, seattle)
    top_words_sf = []
    top_words_seattle = []

    for i in range(len(p0_vec)):
        if p0_vec[i] > -6.0:
            top_words_sf.append((vocab_list[i], p0_vec[i]))
        if p1_vec[i] > -6.0:
            top_words_seattle.append((vocab_list[i], p1_vec[i]))

    sorted_sf = sorted(top_words_sf, key=lambda pair: pair[1], reverse=True)
    print("Top words in SF")
    for item in sorted_sf[:10]:
        print(item[0])

    sorted_seattle = sorted(
        top_words_seattle, key=lambda pair: pair[1], reverse=True)
    print("Top words in Seattle")
    for item in sorted_seattle[:10]:
        print(item[0])
