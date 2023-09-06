from typing import Union

import torch
from monai.transforms import Transform
from omegaconf import ListConfig


class ContrastAdjust(Transform):
    """Transform for contrast adjusting intensity values to be within a range and everything
    outside the range be set to a background value."""

    def __init__(self, low: int, high: int, background: int = 0):
        """
        Parameters
        ----------
        low: int
            lower bound for clipping
        high: int
            upper bound for clipping
        background: int
            intensity value for everywhere outside the specified range
        """
        super().__init__()
        self.low = low
        self.high = high
        self.background = background

    def __call__(self, img):
        low = self.low
        high = self.high
        img = torch.where(img < high, img, self.background)
        img = torch.where(img > low, img, self.background)
        if len(img.shape) < 4:
            img = img.unsqueeze(dim=0)

        return img


class ContrastAdjustd(Transform):
    """Dictionary Transform for clipping image intensities based on absolute or percentile
    values."""

    def __init__(
        self,
        keys: str,
        low: int,
        high: int,
        background: int,
        allow_missing_keys: bool = False,
    ):
        """
        Parameters
        ----------
        keys: str
            name of images to resize

        low: float
            lower bound for clipping
        high: float
            upper bound for clipping
        percentile: bool
            whether to use percentile or absolute values  for clipping
        allow_missing_keys: bool
            whether to fail if provided keys are missing
        """
        super().__init__()
        self.keys = keys if isinstance(keys, (list, ListConfig)) else [keys]
        self.allow_missing_keys = allow_missing_keys
        self.clipper = ContrastAdjust(low, high, background)

    def __call__(self, img_dict):
        for key in self.keys:
            if key not in img_dict.keys() and not self.allow_missing_keys:
                raise KeyError(f"Key {key} not in img_dict")
            img_dict[key] = self.clipper(img_dict[key])
        return img_dict
