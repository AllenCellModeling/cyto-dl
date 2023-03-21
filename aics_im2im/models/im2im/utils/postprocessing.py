from typing import Dict

import numpy as np
import torch
from skimage.exposure import rescale_intensity
from skimage.measure import label


class ActThreshLabel:
    def __init__(
        self, activation=torch.nn.Identity(), threshold=None, label=False, dtype=np.uint8, ch=0
    ):
        self.activation = activation
        self.threshold = threshold
        self.label = label
        self.dtype = dtype
        self.ch = ch

    def __call__(self, img: torch.Tensor) -> np.ndarray:
        img = self.activation(img[self.ch].detach().cpu()).numpy()
        if self.threshold is not None:
            img = img > self.threshold
        if self.label:
            img = label(img)
        return img.astype(self.dtype)


class Rescale:
    def __init__(self, output_dtype):
        self.output_dtype = output_dtype

    def __call__(self, img: torch.Tensor) -> np.ndarray:
        img = detach(img)
        return rescale_intensity(img, out_range=self.output_dtype).astype(self.output_dtype)


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
