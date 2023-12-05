from typing import Sequence

import numpy as np
from monai.transforms import Transform
from omegaconf import ListConfig
from skimage.segmentation import relabel_sequential


class Relabel(Transform):
    def __init__(
        self,
    ):
        super().__init__()

    def __call__(self, img):
        relabeled, _, _ = relabel_sequential(img.numpy().astype(np.int16))
        return relabeled


class Relabeld(Transform):
    """Save a batch of images to disk for debugging."""

    def __init__(
        self,
        keys: Sequence[str],
        allow_missing_keys: bool = False,
    ):
        """
        Parameters
        ----------
        keys: Sequence[str]
            keys to save
        allow_missing_keys: bool
            allow missing keys in batch
        """
        super().__init__()
        self.keys = keys if isinstance(keys, (list, ListConfig)) else [keys]
        self.allow_missing_keys = allow_missing_keys
        self.relabeler = Relabel()

    def __call__(self, img_dict):
        for key in self.keys:
            if key in img_dict:
                img_dict[key] = self.relabeler(img_dict[key])
            elif not self.allow_missing_keys:
                raise ValueError(f"key {key} found in data. Available keys are {img_dict.keys()}")
        return img_dict
