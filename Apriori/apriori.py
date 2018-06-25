def load_data_set():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def gen_set_list_size_one(data_set):
    """ Get a list of itemsets with length 1. """
    itemsets = []
    for trans in data_set:
        for item in trans:
            if [item] not in itemsets:
                itemsets.append([item])
    itemsets.sort()
    return list(map(frozenset, itemsets))


def gen_set_list_size_k(set_list, k):
    """ Grow set size by 1 each time. """
    set_list_size_k = []
    list_size = len(set_list)

    for i in range(list_size):
        for j in range(i + 1, list_size):
            list_1 = list(set_list[i])[:k - 2]
            list_2 = list(set_list[j])[:k - 2]
            list_1.sort()
            list_2.sort()
            if list_1 == list_2:
                set_list_size_k.append(set_list[i] | set_list[j])
    return set_list_size_k


def filter(data_set, set_list, min_support):
    """ Filter out sets that doesn't meet min support. """

    st_dict = {}
    for transaction in data_set:
        for st in set_list:
            if st.issubset(transaction):
                if st not in st_dict:
                    st_dict[st] = 1
                else:
                    st_dict[st] += 1

    num_items = float(len(data_set))
    filtered_set_list = []
    filtered_set_dict = {}

    for st in st_dict:
        support = st_dict[st] / num_items
        if support >= min_support:
            filtered_set_list.insert(0, st)
        filtered_set_dict[st] = support
    return filtered_set_list, filtered_set_dict


def apriori(data_set, min_support=0.5):
    set_list_size_one = gen_set_list_size_one(
        data_set)  # list of sets of length 1
    d = list(map(set, data_set))  # make sets immutable

    f_set_list, f_set_dict = filter(d, set_list_size_one, min_support)
    set_list = [f_set_list]
    k = 2

    while (len(set_list[k - 2]) > 0):
        set_list_size_k = gen_set_list_size_k(set_list[k - 2], k)
        f_set_list_k, f_set_dict_k = filter(d, set_list_size_k, min_support)
        f_set_dict.update(f_set_dict_k)
        set_list.append(f_set_list_k)
        k += 1
    return set_list, f_set_dict
