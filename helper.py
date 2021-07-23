#!/usr/bin/env python
# coding=utf-8

import sys
import numpy as np
from scipy.stats import percentileofscore
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn import preprocessing


def to_percentile(RDM):
    """
    Rescale the values in RDM to percentile values

    Parameters
    ----------
    RDM : ndarray (n_stim, n_stim)
          A representational dissimilarity matrix.

    Returns
    -------
    percentile_RDM: ndarray (n_stim, n_stim)
                    A RDM rescaled to percentile values

    """

    # Determine if RDM has exactly 2 dimensions
    RDM_dim = len(np.shape(RDM))
    if RDM_dim == 2:
        a, b = np.shape(RDM)
    else:
        print("\nThe shape of the RDM should be (n_stim, n_stim).\n")
        return None

    # Determine if RDM is square
    if a != b:
        print("\nThe shape of the RDM should be (n_stim, n_stim).\n")

    # sanitize the RDM
    RDM = np.tril(RDM) + np.triu(RDM.T, 1)
    RDM = np.nan_to_num(RDM)

    # Sort RDM
    sorted_RDM = sorted(np.triu(RDM).reshape(-1, 1))

    # Function that get the percentile of x
    get_percentile = lambda x: percentileofscore(sorted_RDM, x)
    percentile_RDM = get_percentile(RDM)

    return percentile_RDM

def rescale(RDM, vmin=0, vmax=1):
    """
    Rescale the values in RDM to vmin <= value <= vmax

    Parameters
    ----------
    RDM : ndarray (n_stim, n_stim)
          A representational dissimilarity matrix.
    vmin : float Default 0
           The range lower bound.
    vmax : float Default 1
           The range upper bound.

    Returns
    -------
    scaled_RDM: ndarray (n_stim, n_stim)
                A RDM rescaled to vmin <= value <= vmax, with the original RDM
                diagonal values preserved.

    """

    # Determine if RDM has exactly 2 dimensions
    RDM_dim = len(np.shape(RDM))
    if RDM_dim == 2:
        a, b = np.shape(RDM)
    else:
        print("\nThe shape of the RDM should be (n_stim, n_stim).\n")
        return None

    # Determine if RDM is square
    if a != b:
        print("\nThe shape of the RDM should be (n_stim, n_stim).\n")

    # Determine if RDM is square
    if vmax <= vmin:
        print("\nThe upper bound must be strictly bigger than lower bound.\n")

    # Sanitize the RDM
    RDM = np.tril(RDM) + np.triu(RDM.T, 1)
    RDM = np.nan_to_num(RDM)

    scaler = preprocessing.MinMaxScaler(feature_range=(vmin, vmax))
    scaled_RDM = scaler.fit_transform(np.tril(RDM))
    idx_diag = np.diag_indices(a)

    # Sanitize the RDM
    scaled_RDM = scaled_RDM + np.triu(scaled_RDM.T, 1)
    scaled_RDM[idx_diag] = RDM[idx_diag]

    return scaled_RDM


def permutation_test(x, c, metric='mean', n_iter=10000):

    """
    Conduct Permutation test

    Parameters
    ----------
    x : ndarray
        treatment group.
    c : ndarray
        control group.
    metric: string Default 'mean'
            The metric to calculate the similarities. If metric='mean', test the
            mean significance, if metric='spearman', test the Spearman
            correlation significance, if metric='pearson', test the Pearson
            correlation significance, and if metric='kendall', test the Kendall
            tau correlation significance.
    n_iter : int Default 1000.
             Number of iterations.

    Returns
    -------
    p : float
        The permutation test result, p-value.
    """

    x_dim = len(np.shape(x))
    c_dim = len(np.shape(c))
    if x_dim == 1 and c_dim == 1:
        x_len = len(x)
        c_len = len(c)
    else:
        print("\nThe treatment and control groups' samples have to be 1D .\n")
        return None

    # Similarity metrics
    metrics = {'mean': lambda x, c: abs(np.mean(x) - np.mean(c)),
               'spearman': lambda x, c: spearmanr(x, c)[0],
               'pearson': lambda x, c: pearsonr(x, c)[0],
               'kendall': lambda x, c: kendalltau(x, c)[0]}

    # Measured similarity
    measured_metric = metrics[metric](x,c)

    pool = np.hstack([x, c])

    # Permutation test
    perm_metrics = []
    for i in range(n_iter):
        shuffled_pool = np.random.permutation(pool)
        perm_x = shuffled_pool[:x_len]
        perm_c = shuffled_pool[-c_len:]
        perm_metric = metrics[metric](perm_x, perm_c)
        perm_metrics.append(perm_metrics)
        print(i)

    # How many times we observed a value smaller than our measured value?
    votes_count = len(np.where(np.array(perm_metrics) < measured_metric))

    p = 1.0 - ( float(votes_count)/float(n_iter) )

    return p

def show_progressbar(string, current, total=100):

    percent = '{:.2%}'.format(current / total)
    sys.stdout.write('\r')
    sys.stdout.write(string + ": [%-100s] %s" % ('=' * int(current), percent))
    sys.stdout.flush()

def create_coding_model_RDM(stim_mask):
    coding_model_RDM = np.ones()

