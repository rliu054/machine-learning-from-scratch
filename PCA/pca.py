import numpy as np


def load_data_set(file_name, delim='\t'):
    f_input = open(file_name)
    str_arr = [line.strip().split(delim) for line in f_input.readlines()]
    dat_arr = [list(map(float, line)) for line in str_arr]
    return np.mat(dat_arr)


def pca(data_mat, n=9999):
    mean_vals = np.mean(data_mat, axis=0)
    data_mat = data_mat - mean_vals

    cov_mat = np.cov(data_mat, rowvar=0)
    eig_vals, eig_vecs = np.linalg.eig(np.mat(cov_mat))

    eig_val_idx = np.argsort(eig_vals)[:-(n+1):-1]
    new_eig_vecs = eig_vecs[:, eig_val_idx]
    low_d_mat = data_mat * new_eig_vecs
    recon_mat = (low_d_mat * new_eig_vecs.T) + mean_vals

    return low_d_mat, recon_mat
