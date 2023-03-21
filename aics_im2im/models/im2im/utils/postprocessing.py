from typing import Dict

import numpy as np
import torch
from hydra.utils import get_class
from skimage.exposure import rescale_intensity
from skimage.measure import label


class ActThreshLabel:
    def __init__(
        self,
        activation=torch.nn.Identity(),
        threshold=None,
        label=False,
        dtype=np.uint8,
        ch=0,
        rescale_dtype=None,
    ):
        self.activation = activation
        self.threshold = threshold
        self.label = label
        self.dtype = dtype
        self.ch = ch
        self.rescale_dtype = get_class(rescale_dtype)

    def __call__(self, img: torch.Tensor) -> np.ndarray:
        img = self.activation(img[self.ch].detach().cpu().float()).numpy()
        if self.threshold is not None:
            img = img > self.threshold
        if self.label:
            img = label(img)
        if self.rescale_dtype is not None:
            img = rescale_intensity(img, out_range=self.rescale_dtype).astype(self.rescale_dtype)
        return img.astype(self.dtype)


class DictToIm:
    """Convert dictionary with image values to multichannel image."""

    def __init__(self, keys, allow_missing_keys=False):
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        output_img = []
        for key in self.keys:
            if key in input_dict:
                im = detach(input_dict[key]).astype(np.uint8)
                output_img.append(im)
            elif not self.allow_missing_keys:
                raise KeyError(
                    f"key `{key}` not available. Available keys are {input_dict.keys()}"
                )
        return output_img


def detach(img: torch.Tensor) -> np.ndarray:
    img = img.detach().cpu()
    if img.dtype == torch.bfloat16:
        img = img.half()
    img = img.numpy()
    return img
