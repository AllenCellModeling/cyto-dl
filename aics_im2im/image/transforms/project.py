from functools import partial

import torch
from monai.transforms import Transform


class MaxProjectd(Transform):
    def __init__(self, keys, projection_dim=1, allow_missing_keys=False):
        """Dictionary-transform version of SO2RandomRotate."""
        super().__init__()
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.projection_fn = partial(torch.max, dim=projection_dim)

    def __call__(self, input_dict):
        for key in self.keys:
            if key in input_dict.keys():
                input_dict[key], _ = self.projection_fn(input_dict[key])
            elif not self.allow_missing_keys:
                raise KeyError(
                    f"key `{key}` not available. Available keys are {input_dict.keys()}"
                )
        return input_dict
