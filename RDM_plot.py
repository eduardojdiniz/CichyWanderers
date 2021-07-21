#!/usr/bin/env python
# coding=utf-8

# A module for plotting RDMs

import numpy as np
import matplotlib.pyplot as plt


def plot_rdm(rdm, percentile=False, rescale=False, lim=[0, 1], stimuli=None,
             stim_fontsize=12, cmap=None):
    """
    Plot the RDM

    Parameters
    ----------
    rdm : array or list [n_stim, n_stim]
          A representational dissimilarity matrix.
    percentile : bool, Default False.
                 Rescale the values in RDM or not by displaying the percentile.
    rescale : bool, Default False.
              Rescale the values in RDM or not.
              Here, the maximum-minimum method is used to rescale the values
              except for the values on the diagnal.
    lim : array or list [min, max], Default [0, 1].
          The corrs view lims.
    stimuli : string-array or string-list, Default None.
              The labels of the stimuli for plotting. Stimuli should contain
              n_stim strings. If stimuli=None, the labels of conditions will
              be invisible.
    stim_fontsize : int or float, Default 12.
                    The fontsize of the labels of the stimuli for plotting.
    cmap : matplotlib colormap. Default None.
           The colormap for RDM. If cmap=None, the colormap will be 'viridis'.
    """

    # determine if it has 2 dimensions and if it's a square
    a, b = np.shape(rdm)
    if len(np.shape(rdm)) != 2 or a != b:
        print("Invalid input!")
        return None

    # get the number of conditions
    n_stim = rdm.shape[0]

    # if n_stim=2, the RDM cannot be plotted.
    if n_stim == 2:
        print("The shape of RDM cannot be 2*2, we cannot plot this RDM.")
        return None

    if percentile:
        v = np.zeros([n_stim * n_stim, 2], dtype=np.float)
        for i in range(n_stim):
            for j in range(n_stim):
                v[i * n_stim + j, 0] = rdm[i, j]

        index = np.argsort(v[:, 0])
        m = 0
        for i in range(n_stim * n_stim):
            if i > 0:
                if v[index[i], 0] > v[index[i - 1], 0]:
                    m = m + 1
                v[index[i], 1] = m

        v[:, 0] = v[:, 1] * 100 / m

        for i in range(n_stim):
            for j in range(n_stim):
                rdm[i, j] = v[i * n_stim + j, 0]

        if cmap is None:
            plt.imshow(rdm, extent=(0, 1, 0, 1), cmap=plt.get_cmap('viridis'),
                       clim=(0, 100))
        else:
            plt.imshow(rdm, extent=(0, 1, 0, 1), cmap=plt.get_cmap(cmap),
                       clim=(0, 100))

    # rescale the RDM
    elif rescale:
        # flatten the RDM
        vrdm = np.reshape(rdm, [n_stim * n_stim])
        # array -> set -> list
        svrdm = set(vrdm)
        lvrdm = sorted(svrdm)

        # get max & min
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]

        # rescale
        if maxvalue != minvalue:
            for i in range(n_stim):
                for j in range(n_stim):
                    # not on the diagnal
                    if i != j:
                        rdm[i, j] = float(
                            (rdm[i, j] - minvalue) / (maxvalue - minvalue))

        # plot the RDM
        vmin = lim[0]
        vmax = lim[1]
        if cmap is None:
            plt.imshow(rdm, extent=(0, 1, 0, 1), cmap=plt.get_cmap('viridis'),
                       clim=(vmin, vmax))
        else:
            plt.imshow(rdm, extent=(0, 1, 0, 1), cmap=plt.get_cmap(cmap),
                       clim=(vmin, vmax))

    else:
        # plot the RDM
        vmin = lim[0]
        vmax = lim[1]
        if cmap is None:
            plt.imshow(rdm, extent=(0, 1, 0, 1), cmap=plt.get_cmap('viridis'),
                       clim=(vmin, vmax))
        else:
            plt.imshow(rdm, extent=(0, 1, 0, 1), cmap=plt.get_cmap(cmap),
                       clim=(vmin, vmax))

    # plt.axis("off")
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    font = {'size': 18}

    if percentile:
        cb.set_label("Dissimilarity (percentile)", fontdict=font)
    elif rescale:
        cb.set_label("Dissimilarity (Rescaling)", fontdict=font)
    else:
        cb.set_label("Dissimilarity", fontdict=font)

    if stimuli is None:
        plt.axis("off")
    else:
        print("1")
        step = float(1 / n_stim)
        x = np.arange(0.5 * step, 1 + 0.5 * step, step)
        y = np.arange(1 - 0.5 * step, -0.5 * step, -step)
        plt.xticks(x, stimuli, fontsize=stim_fontsize, rotation=30, ha="right")
        plt.yticks(y, stimuli, fontsize=stim_fontsize)

    plt.show()

    return 0


def plot_rdm_withvalue(rdm, lim=[0, 1], value_fontsize=10, stimuli=None,
                       stim_fontsize=12, cmap=None):
    """
    Plot the RDM with values

    Parameters
    ----------
    rdm : array or list [n_stim, n_stim]
          A representational dissimilarity matrix.
    lim : array or list [min, max], Default [0, 1].
          The corrs view lims.
    stim_fontsize : int or float, Default 10.
                    The fontsize of the values on the RDM.
    stimuli : string-array or string-list, Default None.
              The labels of the stimuli for plotting. Stimuli should contain
              n_stim strings. If stimuli=None, the labels of conditions will
              be invisible.
    stim_fontsize : int or float, Default 12.
                    The fontsize of the labels of the stimuli for plotting.
    cmap : matplotlib colormap. Default None.
           The colormap for RDM. If cmap=None, the colormap will be 'Greens'.
    """

    # determine if it has 2 dimensions and if it's a square
    a, b = np.shape(rdm)
    if len(np.shape(rdm)) != 2 or a != b:
        print("Invalid input!")
        return None

    # get the number of conditions
    n_stim = rdm.shape[0]

    # if n_stim=2, the RDM cannot be plotted.
    if n_stim == 2:
        print("The shape of RDM cannot be 2*2, we cannot plot this RDM.")
        return None

    # plot the RDM
    vmin = lim[0]
    vmax = lim[1]
    if cmap is None:
        plt.imshow(rdm, extent=(0, 1, 0, 1), cmap=plt.get_cmap('Greens'),
                   clim=(vmin, vmax))
    else:
        plt.imshow(rdm, extent=(0, 1, 0, 1), cmap=plt.get_cmap(cmap),
                   clim=(vmin, vmax))

    # plt.axis("off")
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    font = {'size': 18}
    cb.set_label("Dissimilarity", fontdict=font)

    # add values
    step = float(1 / n_stim)
    for i in range(n_stim):
        for j in range(n_stim):
            print(i, j)
            _ = plt.text(i * step + 0.5 * step, 1 - j * step - 0.5 * step,
                         float('%.4f' % rdm[i, j]), ha="center", va="center",
                         color="blue", fontsize=value_fontsize)

    if stimuli is None:
        plt.axis("off")
    else:
        print("1")
        step = float(1 / n_stim)
        x = np.arange(0.5 * step, 1 + 0.5 * step, step)
        y = np.arange(1 - 0.5 * step, -0.5 * step, -step)
        plt.xticks(x, stimuli, fontsize=stim_fontsize, rotation=30, ha="right")
        plt.yticks(y, stimuli, fontsize=stim_fontsize)

    plt.show()

    return 0
