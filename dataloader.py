#!/usr/bin/env python
# coding=utf-8

# Imports
import h5py
import scipy.io as sio
import os
import requests
import zipfile
import numpy as np
import glob
import shutil
import pickle


def loadmat(matfile):
    """Function to load .mat files.

    Parameters
    ----------
    matfile : str
    path to `matfile` containing fMRI data for a given trial.

    Returns
    -------
    dict
    dictionary containing data in key 'vol' for a given trial.

    """
    try:
        f = h5py.File(matfile)
    except (IOError, OSError):
        return sio.loadmat(matfile)
    else:
        return {name: np.transpose(f.get(name)) for name in f.keys()}


def download_Cichy(**kwargs):
    """Function to download data from Cichy et al, 2014.

    Parameters
    ----------
    kwargs: dict
    'data_dirpath': str, data directory path. Default: ./data
    'data_url'    : str, data url. Default: https://osf.io/7vpyh/download'
    'label_url'   : str, visual stimuli labels url. Default:
        http://wednesday.csail.mit.edu/MEG1_MEG_Clear_Data/visual_stimuli.mat

    Returns
    -------
    path_dict: dict
    'fMRI'      : str, fMRI filepath
    'MEG'       : str, MEG filepath
    'label'     : str, visual stimuli filepath
    'image'     : str, jpeg images dirpath
    'data'      : str, data dirpath
    'tmp'       : str, temporary data dirpath

    """
    cwd = os.getcwd()
    data_dirpath = kwargs.pop('data_dirpath', os.path.join(cwd, "data"))
    if not os.path.exists(data_dirpath):
        os.makedirs(data_dirpath)

    tmp_dirpath = os.path.join(cwd, "tmp")
    if not os.path.exists(tmp_dirpath):
        os.makedirs(tmp_dirpath)

    data_url = kwargs.pop('data_url', 'https://osf.io/7vpyh/download')
    label_url = kwargs.pop(
        'label_url',
        'http://wednesday.csail.mit.edu/MEG1_MEG_Clear_Data/visual_stimuli.mat')
    data_filepath = os.path.join(tmp_dirpath, 'data.zip')
    label_filepath = os.path.join(tmp_dirpath, 'visual_stimuli.mat')
    if not os.path.exists(tmp_dirpath):
        os.makedirs(tmp_dirpath)

    # Download MEG and fMRI RDMs
    if not os.path.exists(data_filepath):
        r = requests.get(data_url)
        with open(data_filepath, 'wb') as f:
            f.write(r.content)

    # Download visual stimuli
    if not os.path.exists(label_filepath):
        r = requests.get(label_url)
        with open(label_filepath, 'wb') as f:
            f.write(r.content)

    # Extract directory '92_Image_Set' and 'MEG_decoding_RDMs.mat'
    tmp_image_dirpath = os.path.join(tmp_dirpath, '92_Image_Set', '92images')
    if not os.path.exists(tmp_image_dirpath):
        with zipfile.ZipFile(data_filepath, 'r') as zip_file:
            zip_file.extractall(tmp_dirpath)

    # Move image files to permanent directory
    image_dirpath = os.path.join(cwd, 'data', 'images')
    if not os.path.exists(image_dirpath):
        os.makedirs(image_dirpath)
        for f in os.listdir(tmp_image_dirpath):
            shutil.copy(os.path.join(tmp_image_dirpath, f), image_dirpath)

    path_dict = {}
    fMRI_filepath = os.path.join(
        tmp_dirpath,
        '92_Image_Set',
        'target_fmri.mat')
    path_dict['fMRI'] = fMRI_filepath
    path_dict['MEG'] = os.path.join(tmp_dirpath, 'MEG_decoding_RDMs.mat')
    path_dict['label'] = label_filepath
    path_dict['image'] = image_dirpath
    path_dict['data'] = data_dirpath
    path_dict['tmp'] = tmp_dirpath

    return path_dict


def get_stim_dict(**kwargs):
    """Get category names and binary features describing the Cichy dataset

    Parameters
    ----------
    kwargs: dict
    'fMRI'      : str, fMRI filepath
    'MEG'       : str, MEG filepath
    'label'     : str, visual stimuli filepath
    'image'     : str, jpeg images dirpath
    'data'      : str, data dirpath
    'tmp'       : str, temporary data dirpath

    Returns
    -------
    stim_dict: dict
    'category'  : list[str], indicating category
    'human'     : list[int], indicating membership (0=not a member, 1=member)
    'face'      : list[int], indicating membership (0=not a member, 1=member)
    'animate'   : list[int], indicating membership (0=not a member, 1=member)
    'natural'   : list[int], indicating membership (0=not a member, 1=member)
    'imagepath' : list[str], jpeg image filepath

    """
    stimuli_filepath = kwargs.pop('label', '')
    image_dirpath = kwargs.pop('image', '')

    stim_dat = loadmat(stimuli_filepath)['visual_stimuli']
    fields = ['category', 'human', 'face', 'animate', 'natural']

    stim_dict = {field: [] for field in fields}
    for ii in range(92):
        for jj, field in enumerate(fields):
            stim_dict[field].append(stim_dat[0, ii][jj][0])
    for field in fields[1:]:
        stim_dict[field] = np.array(stim_dict[field]).squeeze()
    stim_dict['imagepath'] = sorted(glob.glob(image_dirpath + '/*.jpg'))

    return stim_dict


def get_RDM_dict(**kwargs):
    """Get MEG and fMRI RDMs from the Cichy dataset

    Parameters
    ----------
    kwargs: dict
    'fMRI'      : str, fMRI filepath
    'MEG'       : str, MEG filepath
    'label'     : str, visual stimuli filepath
    'image'     : str, jpeg images dirpath
    'data'      : str, data dirpath
    'tmp'       : str, temporary data dirpath

    Returns
    -------
    RDM_dict: dict
    'MEG'     : ndarray, (16, 2, 1301, 92, 92)
        16 subjects, 2 sessions, 1301 time points (from -100 ms to 1200 ms
        wrt to stimulus onset at 0 ms), 92 conditions by 92 conditions.
        The last 2 dimensions form representational dissimilarity matrices of
        decoding accuracies, symmetric accross the diagonal, with the diagonal
        undefined (NaN).
    'fMRI_EVC': ndarray, (15, 92, 92)
        15 subjects, 92 conditions by 92 conditions.
        The last 2 dimensions form a representational dissimilarity matrix of
        spearman correlation for the EVC cortex, symmetric accross the diagonal,
        with the diagonal undefined (NaN).
    'fMRI_IT' : ndarray, (15, 92, 92)
        15 subjects, 92 conditions by 92 conditions.
        The last 2 dimensions form a representational dissimilarity matrix of
        spearman correlation for the IT cortex, symmetric accross the diagonal,
        with the diagonal undefined (NaN).

    """
    fMRI_filepath = kwargs.pop('fMRI', '')
    MEG_filepath = kwargs.pop('MEG', '')

    RDM_dict = {}

    RDM_dict['MEG'] = loadmat(MEG_filepath)['MEG_decoding_RDMs']

    fMRI_RDMs = loadmat(fMRI_filepath)
    RDM_dict['fMRI_EVC'] = fMRI_RDMs['EVC_RDMs']
    RDM_dict['fMRI_IT'] = fMRI_RDMs['IT_RDMs']

    return RDM_dict


def save_dataset(**kwargs):
    """Save RDM_dict and stim_dict

    Parameters
    ----------
    kwargs: dict
    'fMRI' : str, fMRI filepath
    'MEG'  : str, MEG filepath
    'label': str, visual stimuli filepath
    'image': str, jpeg images dirpath
    'data' : str, data dirpath
    'tmp'  : str, temporary data dirpath
    'RDM'  : dict, see help(get_RDM_dict)
    'stim' : dict, see help(get_stim_dict)

    Returns
    -------
    path_dict: dict
    'RDM.pickle'  : str, RDM dict, pickled
    'stim.pickle' : str, stim dict, pickled
    'label'       : str, visual stimuli filepath
    'image'       : str, jpeg images dirpath
    'data'        : str, data dirpath
    'tmp'         : str, temporary data dirpath

    """

    RDM_dict = kwargs.pop('RDM', '')
    stim_dict = kwargs.pop('stim', '')
    path_dict = kwargs.copy()
    data_dirpath = path_dict['data']

    stim_dict_pickle = os.path.join(data_dirpath, 'stim_dict.pkl')
    RDM_dict_pickle = os.path.join(data_dirpath, 'RDM_dict.pkl')

    with open(stim_dict_pickle, 'wb') as pkl:
        pickle.dump(stim_dict, pkl)
        path_dict['stim.pickle'] = RDM_dict_pickle

    with open(RDM_dict_pickle, 'wb') as pkl:
        pickle.dump(RDM_dict, pkl)
        path_dict['RDM.pickle'] = RDM_dict_pickle

    return path_dict


def load_dataset(**kwargs):
    """Load RDM_dict and stim_dict

    Parameters
    ----------
    path_dict: dict
    'RDM.pickle'  : str, RDM dict, pickled (if save=True)
    'stim.pickle' : str, stim dict, pickled (if save=True)
    'image'       : str, jpeg images dirpath
    'data'        : str, data dirpath

    Returns
    -------
    path_dict: dict
    'RDM.pickle'  : str, RDM dict, pickled (if save=True)
    'stim.pickle' : str, stim dict, pickled (if save=True)
    'image'       : str, jpeg images dirpath
    'data'        : str, data dirpath

    RDM_dict  : dict, see help(get_RDM_dict)

    stim_dict : dict, see help(get_stim_dict)

    """

    path_dict = {}

    # Get data directory path
    data_dirpath = kwargs.pop('data', './data')
    path_dict['data'] = data_dirpath

    # Load stimuli dictionary
    stim_dict_pickle = kwargs.pop('stim.pickle', None)
    if stim_dict_pickle is None:
        stim_dict_pickle = os.path.join(data_dirpath, 'stim_dict.pkl')

    # Load RDM dictionary
    RDM_dict_pickle = kwargs.pop('RDM.pickle', None)
    if RDM_dict_pickle is None:
        RDM_dict_pickle = os.path.join(data_dirpath, 'RDM_dict.pkl')

    with open(stim_dict_pickle, 'rb') as pkl:
        stim_dict = pickle.load(pkl)
        path_dict['stim.pickle'] = stim_dict_pickle

    with open(RDM_dict_pickle, 'rb') as pkl:
        RDM_dict = pickle.load(pkl)
        path_dict['RDM.pickle'] = RDM_dict_pickle

    # Set image directory path
    path_dict = kwargs.pop('image', os.path.join(data_dirpath, 'images'))

    return path_dict, RDM_dict, stim_dict


def create_dataset(save=True, clean=False):
    """Download and organize Cichy et al, 2014 dataset

    Parameters
    ----------
    save_dataset: bool, if True, pickle RDM_dict and stim_dict

    Returns
    -------
    path_dict: dict
    'RDM.pickle'  : str, RDM dict, pickled (if save=True)
    'stim.pickle' : str, stim dict, pickled (if save=True)
    'image'       : str, jpeg images dirpath
    'data'        : str, data dirpath

    RDM_dict  : dict, see help(get_RDM_dict)

    stim_dict : dict, see help(get_stim_dict)

    """

    url_dict = {
        'data_url': 'https://osf.io/7vpyh/download',
        'label_url': 'http://wednesday.csail.mit.edu/MEG1_MEG_Clear_Data/visual_stimuli.mat'}

    # Download Cichy et al, 2014 dataset
    path_dict = download_Cichy(**url_dict)
    # Get stimuli dictionary
    stim_dict = get_stim_dict(**path_dict)
    # Get RDM dictionary
    RDM_dict = get_RDM_dict(**path_dict)

    # Pickle RDM and stimuli dictionaries
    if save:
        data_dict = {'RDM': RDM_dict, 'stim': stim_dict}
        path_dict = save_dataset(**path_dict, **data_dict)

    # Clean temporary directory and path_dict
    tmp_dirpath = path_dict.pop('tmp', '')
    _ = path_dict.pop('label', '')
    if clean:
        shutil.rmtree(tmp_dirpath)

    return path_dict, RDM_dict, stim_dict


if __name__ == "__main__":
    data_dirpath, RDM_dict, stim_dict = create_dataset()
    # print(RDM_dict['MEG'].shape)
    # print(stim_dict.keys())
