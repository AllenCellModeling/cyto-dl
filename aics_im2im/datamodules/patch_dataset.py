import sys
from collections import deque
from typing import List, Sequence, Union
import logging

import pandas as pd
import yaml
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from serotiny.io.dataframe import DataframeDataset
from hydra.utils import instantiate
import random
from aics_im2im.utils import MultiScaleCropper

logger = logging.getLogger(__name__)
ArrayLike = Union[np.ndarray, torch.Tensor]

class BufferedPatchDataset(Dataset):
    """
    Provides patches from items of a dataset.

    Parameters
    ----------
    dataset
        Dataset object.

    patch_shape
        Shape of patch to be extracted from dataset items.

    buffer_size
        Size of buffer.

    buffer_switch_interval
        Number of patches provided between buffer item exchanges. Set to -1 to
        disable exchanges.

    shuffle_images
        Set to randomize order of dataset item insertion into buffer.

    """

    def __init__(
        self,
        dataset,
        data_config,
        imgs_per_epoch,
        patches_per_image,
        input_patch_size,
    ):
        self.dataset = dataset
        self.imgs_per_epoch = min(len(self.dataset), imgs_per_epoch)
        self.patch_per_img = patches_per_image
        self.input_patch_size = input_patch_size
        self.patches = {}
        self.patches_loaded = 0
        self.scales = np.unique([data_config[k] for k in data_config.keys()])
        self.data_config = data_config

    def load_patches_from_image(self):
        #transforms happen here
        images = self.dataset[self.images_loaded]
        max_shape = np.max([im.shape for _, im in images.items()])
        cropper = MultiScaleCropper(max_shape, self.input_patch_size, self.scales, self.patches_per_image)
        patches = cropper(images, self.data_config)

        for k, v in patches:
            try:
                self.patches[k] = v
            except KeyError:
                self.patches[k] += v
        self.patches_loaded  += self.patch_per_img

    def __len__(self):        
        return self.imgs_per_epoch * self.patch_per_img

    def __getitem__(self, index):
        while index > self.patches_loaded:
            self.load_patches_from_image()
        else:
            return {k: torch.from_numpy(v[index].float()) for k, v in self.patches.items()}
            
    
