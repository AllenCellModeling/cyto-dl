from typing import Union

import torch
from monai.transforms import Transform
from omegaconf import ListConfig


class MaxProjectd(Transform):
    """Monai-style transform to take max projection of an image."""

    def __init__(
        self,
        keys: Union[list, str],
        projection_dim: int = 1,
        allow_missing_keys: bool = False,
    ):
        """
        Parameters
        ----------
        keys: Union[list, str]
            keys to apply max projection
        projection_dim: int=1
            index into NCZYX to compute projection across
        allow_missing_keys: bool=False
            Whether to raise error if specified key is missing
        """
        super().__init__()
        self.keys = keys if isinstance(keys, (list, ListConfig)) else [keys]
        self.allow_missing_keys = allow_missing_keys
        self.projection_dim = projection_dim

    def __call__(self, input_dict):
        """
        Parameters
        ----------
        input_dict: Dict[str, torch.Tensor]
            dict of CZYX tensors/metadata
        """
        for key in self.keys:
            if key in input_dict.keys():
                input_dict[key], _ = torch.max(input_dict[key], dim=self.projection_dim)
            elif not self.allow_missing_keys:
                raise KeyError(
                    f"key `{key}` not available. Available keys are {input_dict.keys()}"
                )
        return input_dict
