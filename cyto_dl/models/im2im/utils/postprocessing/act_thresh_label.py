from typing import Callable

import numpy as np
import torch
from hydra.utils import get_class
from numpy.typing import DTypeLike
from skimage.exposure import rescale_intensity
from skimage.measure import label


class ActThreshLabel:
    """General-purpose postprocessing transform for applying any of an activation, threshold,
    labeling, and rescaling."""

    def __init__(
        self,
        activation: Callable = torch.nn.Identity(),
        threshold: float = None,
        label: bool = False,
        dtype: DTypeLike = np.uint8,
        ch: int = -1,
        rescale_dtype: DTypeLike = None,
    ):
        """
        Parameters
        ----------
        activation: Callable=torch.nn.Identity()
            activation to apply to image
        threshold: float =None
            Threshold value, defaults to None, where no threshold is applied
        label:bool=False
            whether to label image after thresholding
        dtype:DTypeLike=np.uint8
            data type of output image, defaults to np.uint8
        ch:int=-1
            channel of image to apply postprocessing to, default -1 for all channels
        rescale_dtype=None
            dtype to rescale intensity range to, defaults to no rescaling.
        """
        self.activation = activation
        self.threshold = threshold
        self.label = label
        self.dtype = self._get_dtype(dtype)
        self.ch = ch
        self.rescale_dtype = self._get_dtype(rescale_dtype)
        if self.rescale_dtype is not None:
            self.dtype = self.rescale_dtype

    def _get_dtype(self, dtype: DTypeLike) -> DTypeLike:
        if isinstance(dtype, str):
            return get_class(dtype)
        elif dtype is None:
            return dtype
        elif isinstance(dtype, type):
            return dtype
        else:
            raise ValueError(f"Expected dtype to be DtypeLike, string, or None, got {type(dtype)}")

    def __call__(self, img: torch.Tensor) -> np.ndarray:
        if self.ch > 0:
            img = img[self.ch]
        img = self.activation(img.detach().cpu().float()).numpy()
        if self.threshold is not None:
            img = img > self.threshold
        if self.label:
            img = label(img)
        if self.rescale_dtype is not None:
            img = rescale_intensity(img, out_range=self.rescale_dtype).astype(self.rescale_dtype)
        return img.astype(self.dtype)
