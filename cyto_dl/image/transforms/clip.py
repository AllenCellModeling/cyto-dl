from typing import Union

import torch
from monai.transforms import Transform
from omegaconf import ListConfig


class Clip(Transform):
    """Transform for clipping image intensities based on absolute or percentile values."""

    def __init__(self, low: float = 0.0001, high: float = 0.9999, percentile=True):
        """
        Parameters
        ----------
        low: float
            lower bound for clipping
        high: float
            upper bound for clipping
        percentile: bool
            whether to use percentile or absolute values  for clipping
        """
        super().__init__()
        self.low = low
        self.high = high
        self.percentile = percentile

    def __call__(self, img):
        low = self.low
        high = self.high
        if self.percentile:
            low = torch.quantile(img, self.percentile_low)
            high = torch.quantile(img, self.percentile_high)

        return torch.clip(img, low, high)


class Clipd(Transform):
    """Dictionary Transform for clipping image intensities based on absolute or percentile
    values."""

    def __init__(
        self,
        keys: str,
        low: float = 0.0001,
        high: float = 0.9999,
        percentile=True,
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
        self.clipper = Clip(low, high, percentile)

    def __call__(self, img_dict):
        for key in self.keys:
            if key not in img_dict.keys() and not self.allow_missing_keys:
                raise KeyError(f"Key {key} not in img_dict")
            img_dict[key] = self.clipper(img_dict[key])
        return img_dict
