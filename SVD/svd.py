import numpy as np


def load_data():
    data = [[1, 1, 1, 0, 0], [2, 2, 2, 0, 0], [1, 1, 1, 0, 0], [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2], [0, 0, 0, 3, 3], [0, 0, 0, 1, 1]]
    return data


def euclid_sim(a, b):
    return 1.0 / (1.0 + np.linalg.norm(a - b))


def pears_sim(a, b):
    if len(a) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(a, b, rowvar=0)[0][1]


def cos_sim(a, b):
    num = float(a.T * b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.5 + 0.5 * (num / denom)


def stand_est(data, user, sim_meas, item):
    n = np.shape(data)[1]
    sim_total = 0.0
    rat_sim_total = 0.0

    for j in range(n):
        user_rating = data[user, j]
        if user_rating == 0:
            continue
        overlap = np.nonzero(
            np.logical_and(data[:, item].A > 0, data[:, j].A > 0))[0]
        if len(overlap) == 0:
            similarity = 0
        else:
            similarity = sim_meas(data[overlap, item], data[overlap, j])
        print("%d and %d similarity is: %f" % (item, j, similarity))
        sim_total += similarity
        rat_sim_total += similarity * user_rating
    if sim_total == 0:
        return 0
    else:
        return rat_sim_total / sim_total


def recommend(data, user, n=3, sim_meas=cos_sim, est_method=stand_est):
    unrated_items = np.nonzero(data[user, :].A == 0)[1]
    if len(unrated_items) == 0:
        return "all rated"

    item_scores = []
    for item in unrated_items:
        est_score = est_method(data, user, sim_meas, item)
        item_scores.append((item, est_score))
    return sorted(item_scores, key=lambda jj: jj[1], reverse=True)[:n]


def print_mat(data_mat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(data_mat[i, k]) > thresh:
                print(1, end="")
            else:
                print(0, end="")
        print("")


def img_compress(num_sv=3, thresh=0.8):
    my_l = []
    for line in open('../Data/SVD/0_5.txt').readlines():
        row = []
        for i in range(32):
            row.append(int(line[i]))
        my_l.append(row)
    data_mat = np.mat(my_l)
    print("*** original matrix ***")
    print_mat(data_mat, thresh)

    u, sigma, vt = np.linalg.svd(data_mat)
    sig_recon = np.mat(np.zeros((num_sv, num_sv)))
    for k in range(num_sv):
        sig_recon[k, k] = sigma[k]
    recon_mat = u[:, :num_sv] * sig_recon * vt[:num_sv, :]
    print("*** reconstructed matrix using %d singular values ***" % num_sv)
    print_mat(recon_mat, thresh)
