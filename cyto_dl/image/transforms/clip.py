import torch
from monai.transforms import Transform
from monai.transforms.utils_pytorch_numpy_unification import clip, percentile
from omegaconf import ListConfig


class Clip(Transform):
    """Transform for clipping image intensities based on absolute or percentile values."""

    def __init__(self, low: float = 0.01, high: float = 99.99, percentile=True):
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
            low = percentile(img, low)
            high = percentile(img, high)

        return clip(img, low, high)


class Clipd(Transform):
    """Dictionary Transform for clipping image intensities based on absolute or percentile
    values."""

    def __init__(
        self,
        keys: str,
        low: float = 00.01,
        high: float = 99.99,
        percentile=True,
        allow_missing_keys: bool = False,
        per_channel=True,
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
        self.per_channel = per_channel

    def __call__(self, img_dict):
        for key in self.keys:
            if key in img_dict.keys():
                if self.per_channel:
                    img_dict[key] = torch.stack([self.clipper(img) for img in img_dict[key]])
                else:
                    img_dict[key] = self.clipper(img_dict[key])
            elif not self.allow_missing_keys:
                raise KeyError(f"key `{key}` not available. Available keys are {img_dict.keys()}")
        return img_dict
