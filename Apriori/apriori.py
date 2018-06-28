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
    set_list_size_one = gen_set_list_size_one(data_set)
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


def generate_rules(set_list, set_dict, min_conf=0.7):
    rules_list = []

    for i in range(1, len(set_list)):  # only for sets with two or more items
        for freq_set in set_list[i]:
            print("freq_set: {}".format(freq_set))
            h1 = [frozenset([item]) for item in freq_set]
            if i > 1:
                rules_from_conseq(freq_set, h1, set_dict, rules_list, min_conf)
            else:
                calc_conf(freq_set, h1, set_dict, rules_list, min_conf)


def calc_conf(freq_set, h, set_dict, rules_list, min_conf=0.7):
    print("in calc_conf")

    pruned_h = []
    for conseq in h:
        print("freq_set: {}, conseq: {}\n".format(freq_set, conseq))
        conf = set_dict[freq_set] / set_dict[freq_set - conseq]
        if conf >= min_conf:
            print(freq_set - conseq, '-->', conseq, 'conf', conf)
            rules_list.append((freq_set - conseq, conseq, conf))
            pruned_h.append(conseq)
    return pruned_h


def rules_from_conseq(freq_set, h, set_dict, rules_list, min_conf=0.7):
    m = len(h[0])
    print("in rules from conseq, h={}".format(h))
    print("freq_set={}, m={}".format(freq_set, m))

    if len(freq_set) > m + 1:
        hmp1 = gen_set_list_size_k(h, m + 1)
        print("before, hmp1={}".format(hmp1))
        hmp1 = calc_conf(freq_set, hmp1, set_dict, rules_list, min_conf)
        print("after, hmp1={}".format(hmp1))

        if len(hmp1) > 1:
            rules_from_conseq(freq_set, hmp1, set_dict, rules_list, min_conf)


def print_rules(rules_list, item_meaning):
    for rule in rules_list:
        for item in rule[0]:
            print(item_meaning[item])
        print("    ---->")
        for item in rule[1]:
            print(item_meaning[item])
        print("confidence: %f" % rule[2])
        print()
