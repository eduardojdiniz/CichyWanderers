#!/usr/bin/env python
# coding: utf-8

# Imports
import os
import glob
import numpy as np
import torch
from torch.autograd import Variable as V
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import transforms as trn
import requests
from PIL import Image
from tqdm import tqdm
import pickle

# AlexNet Definition
__all__ = ['AlexNet', 'alexnet']

# Rdefine AlexNet differently from torchvision code for better understanding
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            )
        self.fc6 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            )
        self.fc7 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            )
        self.fc8 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            )
        self.num_layers = 8

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)

        out5_reshaped = out5.view(out5.size(0), 256 * 6 * 6)
        out6 = self.fc6(out5_reshaped)
        out7 = self.fc7(out6)
        out8 = self.fc8(out7)
        return out1, out2, out3, out4, out5, out6, out7, out8

    def get_key_list(self):
        key_list = ["conv1.0.weight", "conv1.0.bias",
                     "conv2.0.weight", "conv2.0.bias",
                     "conv3.0.weight", "conv3.0.bias",
                     "conv4.0.weight", "conv4.0.bias",
                     "conv5.0.weight", "conv5.0.bias",
                     "fc6.1.weight", "fc6.1.bias",
                     "fc7.1.weight", "fc7.1.bias",
                     "fc8.1.weight", "fc8.1.bias"]
        return key_list


def alexnet(pretrained=False, ckpth_urls=None, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Parameters
    ----------
    pretrained (bool): if True, returns AlexNet pre-trained on ImageNet
    'ckpth_urls (dict): key -> model; value -> url to pre-trained weights.
    Default: ckpth_urls['alexnet'] =
        'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'

    kwargs (dict): parameters to AlexNet model Class

    Returns
    -------
    model (AlexNet): Pytorch instance of AlexNet model Class

    """

    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(ckpth_urls['alexnet']))

    return model


def load_alexnet(pretrained=False, custom_keys=False, **kwargs):
    """ This function initializes an Alexnet and load its weights from a
    pretrained model. Since we redefined model in a different way we have to
    rename the weights that were in the pretrained checkpoint.

    Parameters
    ----------
    pretrained (bool) : if True, returns AlexNet pre-trained on ImageNet. Don't
    use if using custom checkpoint/state keys definition.
    custom_keys (bool): if True, returns AlexNet pre-trained on ImageNet using
    using custom checkpoint/state keys definition.
    kwargs (dict)
    'ckpth_urls'      : dict,
    'ckpth'           : str, filepath to pretrained AlexNet checkpoint
    Other entries are parameters to AlexNet


    Returns
    -------
    model (AlexNet): Pytorch instance of the AlexNet model Class

    """

    if pretrained:
        ckpth_urls = kwargs.pop('ckpth_urls', None)
        ckpth_filepath = kwargs.pop('ckpth', './models/alexnet/alexnet.pth')
        if ckpth_urls is None:
            url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
            ckpth_urls = {'alexnet': url}

        if custom_keys:
            # Don't use default state keys for pretrained weights
            model = alexnet(pretrained=False, **kwargs)

            # Download pretrained Alexnet from:
            # https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
            # and save in the model directory
            if not os.path.exists(ckpth_filepath):
                r = requests.get(ckpth_urls['alexnet'])
                with open(ckpth_filepath, 'wb') as f:
                    f.write(r.content)

            ckpth = torch.load(ckpth_filepath,
                               map_location=lambda storage, loc: storage)

            # Remap checkpoint/state keys
            key_list = kwargs.pop('key_list', model.get_key_list())
            state_dict = {key_list[i]: v
                          for i, (k, v) in enumerate(ckpth.items())}

            # initialize model with pretrained weights
            model.load_state_dict(state_dict)

        else:
            model = alexnet(pretrained=True, ckpth_urls=ckpth_urls, **kwargs)
    else:
        model = alexnet(**kwargs)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    return model


def get_alexnet_activations_and_save(model, **kwargs):
    """ Generates Alexnet features and save them in a specified directory.

    Parameters
    ----------
    model (AlexNet): AlexNet Pytorch model
    kwargs (dict)
    'images'       : list[str], contain filepath to all images
    'activations'  : str, save dirpath for extracted features.
    """

    # Resize and normalize transform for images
    resize_normalize = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # for all images in the list generate and save activations
    image_list = kwargs.pop('images', '')
    activations_dirpath = kwargs.pop('activations', '')
    for image_file in tqdm(image_list):
        # open image
        img = Image.open(image_file)
        img_filename = os.path.split(image_file)[-1].split(".")[0]

        # apply transformations before feeding to model
        input_img = V(resize_normalize(img).unsqueeze(0))
        if torch.cuda.is_available():
            input_img = input_img.cuda()
        x = model.forward(input_img)

        activations = []
        for i, feat in enumerate(x):
            activations.append(feat.data.cpu().numpy().ravel())

        for layer in range(len(activations)):
            filename = img_filename + "_layer_" + str(layer+1) + ".npy"
            save_filepath = os.path.join(activations_dirpath, filename)
            np.save(save_filepath, activations[layer])


def get_alexnet_RDM_dict(**kwargs):
    """ Computes RDMs for each layer of an AlexNet model given the layers'
    activations.

    Parameters
    ----------
    num_layers (int): Number of the layers in the AlexNet model. Default: 8

    path_dict (dict)
    'activations' (str): path to the directory with the saved activations. Each
    file in the directory is of type .npy and holds the activations of a
    particular layer for each of the stimuli images.

    Returns
    -------
    alexnet_RDM_dict (dict[ndarray]): Each entry stores a (92, 92) ndarray RDM,
    where the keys are the 8 layers of the AlexNet class model.
    E.g., alexnet_RDM_dict['layer_1'] = ndarray(shape=(92,92))

    """

    # number of layers in the model
    num_layers = kwargs.pop('num_layers', 8)
    activations_dir = kwargs.pop('activations', 8)

    layers = []
    for i in range(num_layers):
      layers.append("layer" + "_" + str(i+1))

    alexnet_RDM_dict = {}

    # create RDM for each layer from activations
    for layer in layers:
        activation_files = glob.glob(activations_dir + '/*' + layer + '.npy')
        activation_files.sort()

        # Load all activations
        activations = []
        for activation_file in activation_files:
            activations.append(np.load(activation_file))
        activations = np.array(activations)

        # calculate Pearson's distance for all pairwise comparisons
        alexnet_RDM_dict[layer] = 1 - np.corrcoef(activations)

    return alexnet_RDM_dict


def save_alexnet_RDM_dict(**kwargs):
    """Save AlexNet RDMs dictionary

    Parameters
    ----------
    kwargs: dict
    'savepath': str, model dirpath, where to save the alexnet_RDM_dict pickle
    'RDM_dict'  : dict, see help(get_alexnet_RDM_dict)

    Returns
    -------
    alexnet_RDM_dict.pkl (str) Alexnet Layers' RDMs dict, pickle path

    """

    alexnet_RDM_dict = kwargs.pop('RDM_dict', '')
    model_dirpath = kwargs.pop('savepath', '')

    alexnet_RDM_dict_pkl = os.path.join(model_dirpath, 'alexnet_RDM_dict.pkl')

    with open(alexnet_RDM_dict_pkl, 'wb') as pkl:
        pickle.dump(alexnet_RDM_dict, pkl)

    return alexnet_RDM_dict_pkl


def load_alexnet_RDM_dict(**kwargs):
    """Load AlexNet RDMs dictionary

    Parameters
    ----------
    alexnet_RDM_dict.pkl (str) Alexnet Layers' RDMs dict, pickle path

    Returns
    -------
    alexnet_RDM_dict' (dict): see help(get_alexnet_RDM_dict)

    """

    # Load RDM dictionary
    alexnet_RDM_dict_pickle = kwargs.pop('alexnet_RDM_dict.pkl', None)

    with open(alexnet_RDM_dict_pickle, 'rb') as pkl:
        alexnet_RDM_dict = pickle.load(pkl)

    return alexnet_RDM_dict


def main(**kwargs):
    """ Creates an instance of AlexNet model class, set it's state to a
    pretrained AlexNet model on ImageNet, get and saves the layers' activations
    for each of the visual stimuli image, and computes the corresponding RDM
    for each layer.

    Parameters
    ----------
    kwargs (dict): parameters to AlexNet model Class

    Returns
    -------
    alexnet_RDM_dict (dict[ndarray]): Each entry stores a (92, 92) ndarray RDM,
    where the keys are the 8 layers of the AlexNet class model.
    E.g., alexnet_RDM_dict['layer_1'] = ndarray(shape=(92,92))

    """

    # load model state from checkpoint
    pretrained = True

    # Remap checkpoint state keys to custom layer names
    custom_keys = True

    cwd = os.getcwd()  # get current working directory

    # Directory where stimuli images are located
    image_dirpath = os.path.join(cwd, 'data', 'images')

    # Directory to save model related data
    alexnet_model_dirpath = os.path.join(cwd, 'models', 'alexnet')
    if not os.path.exists(alexnet_model_dirpath):
        os.makedirs(alexnet_model_dirpath)
    ckpth_filepath = os.path.join(alexnet_model_dirpath, 'alexnet.pth')

    # Alexnet pretrained on ImageNet checkpoint url
    url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
    ckpth_urls = {'alexnet': url}

    alexnet_model_dirpath = os.path.join(cwd, 'models', 'alexnet')

    # load Alexnet initialized with pretrained weights
    model = load_alexnet(pretrained=True, custom_keys=True,
                         ckpth_urls=ckpth_urls, ckpth=ckpth_filepath, **kwargs)

    # get and save activations
    activations_dirpath = os.path.join(alexnet_model_dirpath, 'activations')
    if not os.path.exists(activations_dirpath):
        os.makedirs(activations_dirpath)

    path_dict = {}
    path_dict['activations'] = activations_dirpath

    path_dict['images'] = sorted(glob.glob(image_dirpath + '/*.jpg'))
    get_alexnet_activations_and_save(model, **path_dict)

    # Get AlexNet's layers RDMs
    num_layers = model.num_layers
    alexnet_RDM_dict = get_alexnet_RDM_dict(num_layers=num_layers, **path_dict)
    alexnet_RDM_dict_pkl = save_alexnet_RDM_dict(RDM_dict = alexnet_RDM_dict,
                                               savepath = alexnet_model_dirpath)

    return alexnet_RDM_dict

if __name__ == "__main__":
    alexnet_RDM_dict = main()
