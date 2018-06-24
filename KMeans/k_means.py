import numpy as np


def load_data_set(filename):
    data_mat = []
    input_file = open(filename)
    for line in input_file.readlines():
        cur_line = line.strip().split('\t')
        flt_line = list(map(float, cur_line))  # map all elements to float()
        data_mat.append(flt_line)
    return data_mat


def euclid_dist(vec_a, vec_b):
    """ Euclid distance. """

    return np.sqrt(np.sum(np.power(vec_a - vec_b, 2)))


def rand_cent(data_set, k):
    """ Pick random centroids. """

    n = np.shape(data_set)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        min_j = np.min(data_set[:, j])
        range_j = float(np.max(data_set[:, j]) - min_j)
        centroids[:, j] = np.mat(min_j + range_j * np.random.rand(k, 1))
    return centroids


def k_means(data_set, k, dist_measure=euclid_dist, create_cent=rand_cent):
    m = np.shape(data_set)[0]
    assign_table = np.mat(np.zeros((m, 2)))  # centroid assignments

    centroids = create_cent(data_set, k)
    cluster_updated = True

    while cluster_updated:
        cluster_updated = False
        for i in range(m):
            min_dist = np.inf
            min_idx = -1
            for j in range(k):
                dist_ij = dist_measure(centroids[j, :], data_set[i, :])
                if dist_ij < min_dist:
                    min_dist = dist_ij
                    min_idx = j
            if assign_table[i, 0] != min_idx:
                cluster_updated = True
            assign_table[i, :] = min_idx, min_dist**2
        print(centroids)

        for cent in range(k):
            cluster = data_set[np.nonzero(assign_table[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(cluster, axis=0)

    return centroids, assign_table


def clus_idx(cluster, i, same_cluster=True):
    if same_cluster:
        return np.nonzero(cluster[:, 0].A == i)[0]
    else:
        return np.nonzero(cluster[:, 0].A != i)[0]


def bi_k_means(data_set, k, dist_measure=euclid_dist):
    m = np.shape(data_set)[0]
    clus_assign = np.mat(np.zeros((m, 2)))
    centroid0 = np.mean(data_set, axis=0).tolist()[0]
    cent_list = [centroid0]

    for j in range(m):
        clus_assign[j, 1] = euclid_dist(np.mat(centroid0), data_set[j, :])**2
    while len(cent_list) < k:
        min_sse = np.inf
        for i in range(len(cent_list)):
            cluster = data_set[clus_idx(clus_assign, i), :]
            centroid_mat, split_clus_ass = k_means(cluster, 2, dist_measure)
            sse_split = np.sum(split_clus_ass[:, 1])
            sse_no_sp = np.sum(clus_assign[clus_idx(clus_assign, i, False), 1])
            print("sse_split, and not_split: ", sse_split, sse_no_sp)

            if (sse_split + sse_no_sp) < min_sse:
                best_sp = i
                best_new_cents = centroid_mat
                best_clus_assign = split_clus_ass.copy()
                min_sse = sse_split + sse_no_sp
            best_clus_assign[clus_idx(best_clus_assign, 1), 0] = len(cent_list)
            best_clus_assign[clus_idx(best_clus_assign, 0), 0] = best_sp
            print('the bestCentToSplit is: ', best_sp)
            print('the len of bestClustAss is: ', len(best_clus_assign))

            cent_list[best_sp] = best_new_cents[0, :].tolist()[0]
            cent_list.append(best_new_cents[1, :].tolist()[0])
            clus_assign[clus_idx(clus_assign, best_sp), :] = best_clus_assign
    return np.mat(cent_list), clus_assign
