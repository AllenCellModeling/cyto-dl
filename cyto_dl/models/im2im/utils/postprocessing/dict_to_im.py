from typing import Dict

import numpy
import numpy as np
import torch


def detach(img: torch.Tensor) -> np.ndarray:
    """Convert CUDA tensor to numpy array on cpu."""
    img = img.detach().cpu()
    if img.dtype == torch.bfloat16:
        img = img.half()
    img = img.numpy()
    return img


class DictToIm:
    """Convert dictionary with image values to multichannel image."""

    def __init__(self, keys, allow_missing_keys: bool = False):
        """
        Parameters
        ----------
        keys: Union[str, List[str]]
            keys from dictionary to concatenate into multichannel image
        allow_missing_keys: bool = False
            whether to raise error if specified key is missing
        """
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, input_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        output_img = []
        for key in self.keys:
            if key in input_dict:
                im = detach(input_dict[key]).astype(np.uint8)
                output_img.append(im)
            elif not self.allow_missing_keys:
                raise KeyError(
                    f"key `{key}` not available. Available keys are {input_dict.keys()}"
                )
        return np.stack(output_img)
