#!/usr/bin/env python
# coding=utf-8

import numpy as np
from tqdm import tqdm, trange
from scipy.stats import spearmanr, pearsonr, kendalltau
from helper import rescale, permutation_test


def RSA(model_RDM, data_RDM, metric="spearman", rescale=False,
        permutation=False, n_iter=10000):
    """
    Calculate the similarity between data RDM(s) and a model RDM

    Parameters
    ----------
    model_RDM : ndarray (n_stim, n_stim)
        An model RDM.
    data_RDM : ndarray
               The data RDM(s). The shape can be (n_stim, n_stim) or
               (n_timepoints, n_stim, n_stim) or
               (n_subj, n_timepoints, n_stim, n_stim).
    metric : string 'spearman' or 'pearson' or 'kendall' or 'similarity' or
             'distance' Default 'spearman'.
             The metric to calculate the similarities. If metric='spearman',
             calculate the Spearman Correlations. If metric='pearson',
             calculate the Pearson Correlations. If metric='kendall',
             calculate the Kendall tau Correlations.
    rescale : bool True or False Default False.
              Rescale the values in RDM or not. Here, the maximum-minimum method
              is used to rescale the values except for the ones on the diagonal.
    permutation : bool True or False Default False.
                  Use permutation test or not.
    n_iter : int Default 1000.
             The times for iteration.

    Returns
    -------
    similarities: ndarray
        The similarities between data RDM(s) and an model RDM
        If the shape of data_RDM is [n_stim, n_stim], the shape of similarities
        will be (1,2), a rho-value and a p-value. If the shape of data_RDM is
        (n_timepoints, n_stim, n_stim), the shape of similarities will be
        (n_timepoints, 2). If the shape of data_RDM is
        (n_subjects, n_timepoints, 2), the shape of similarities will be
        (n_subjects, n_timepoints, 2).

    Notes
    -----
    The model RDM could be a behavioral RDM, a hypothesis-based coding model RDM
    or a computational model RDM.

    """

    # Determine if model RDM has exactly 2 dimensions
    model_dim = len(np.shape(model_RDM))
    if model_dim == 2:
        a, b = np.shape(model_RDM)
    else:
        print("\nThe shape of the model RDM should be (n_stim, n_stim).\n")
        return None

    # Determine if model RDM is square
    if a != b:
        print("\nThe shape of the model RDM should be (n_stim, n_stim).\n")
        return None

    data_dim = len(np.shape(data_RDM))
    if data_dim >= 2:
        a, b = np.shape(data_RDM)[-2:]
    else:
        print("\nThe shape of the data RDM should be (n_stim, n_stim) or \
              (n_timepoints, n_stim, n_stim) or (n_subjects, n_timepoins, \
              n_stim, n_stim).\n")
        return None

    if data_dim > 4 or a != b:
        print("\nThe shape of the data RDM should be (n_stim, n_stim) or \
              (n_timepoints, n_stim, n_stim) or (n_subjects, n_timepoins, \
              n_stim, n_stim).\n")
        return None

    # Similarity metrics
    metrics = {'mean': lambda x, c: abs(np.mean(x) - np.mean(c)),
               'spearman': lambda x, c: spearmanr(x, c)[0],
               'pearson': lambda x, c: pearsonr(x, c)[0],
               'kendall': lambda x, c: kendalltau(x, c)[0]}


    model = np.tril(model_RDM).reshape(-1)

    if data_dim == 4:
        n_subjects, n_timepoints = data_RDM.shape[:2]

    if data_dim == 3:
        n_subjects = 1
        n_timepoints = data_RDM.shape[0]
        data_RDM = data_RDM[np.newaxis, :]

    if data_dim == 2:
        n_subjects = 1
        n_timepoints = 1
        data_RDM = data_RDM[np.newaxis, np.newaxis, :]

    similarities = np.zeros([n_subjects, n_timepoints, 2], dtype=np.float64)
    total = n_subjects * n_timepoints

    # Computing the Similarities
    print("\nComputing similarities")
    for i in trange(n_subjects):
        for t in trange(n_timepoints):
            RDM = np.triu(data_RDM[i, t]).reshape(-1)

            if rescale:
                RDM = rescale(RDM)
            similarities[i, t, 0] = metrics[metric](model, RDM)

            if permutation:
                similarities[i, t, 1] = permutation_test(model, RDM,
                                                            metric=metric,
                                                            n_iter=n_iter)

    return similiarities
