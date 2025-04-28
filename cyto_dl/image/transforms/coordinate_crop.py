from typing import List, Optional

import numpy as np
import torch
from monai.transforms import Transform
from omegaconf import ListConfig


class CropToCoordsd(Transform):
    def __init__(
        self,
        keys: List[str],
        start_keys: List[str],
        end_keys: Optional[List[str]] = None,
        crop_size: Optional[int] = None,
        start_is_slice: bool = False,
    ):
        """Crop an image to the given start and end coordinates.

        Parameters
        ----------
        keys: List[str]
            The keys of the images to be cropped.
        start_keys: List[str]
            Names of the keys containing start coordinates. The number of keys passed should be the same as the spatial dimensions of your image (e.g. 2D images should have a `start_y` and a `start_x` key). If start_is_slice is True, the keys should hold a list of slices. Coordinates should be passed as a comma separated string or a list/numpy array/tensor of integers.
        end_keys: List[str], optional
            Names of the keys containing start coordinates. The number of keys passed should be the same as the spatial dimensions of your image (e.g. 2D images should have a `end_y` and a `end_x` key). If end_keys is None, the crop size will be used to calculate the end coordinates. Coordinates should be passed as a comma separated string or a list/numpy array/tensor of integers.
        crop_size: List[int], optional
            The size of the crop. If end_keys is None, this will be used to calculate the end coordinates. Crop size should have the same number of elements as the number of spatial dimensions.
        start_is_slice: bool, optional
            If True, the start coordinates are given as slices. There should only be one start key
        """
        super().__init__()
        self.keys = keys
        if start_is_slice:
            assert (
                len(start_keys) == 1
            ), "If start_is_slice is True, there should only be one start key containing a list of slices"
            start_keys = start_keys[0]
        if crop_size is None and not start_is_slice:
            assert len(start_keys) == len(
                end_keys
            ), "`start_keys` and `end_keys` must have the same length unless `crop_size` is provided or `start_is_slice` is True."

        self.start_keys = start_keys
        self.end_keys = end_keys
        self.crop_size = crop_size

        self.start_is_slice = start_is_slice

    def _get_slice(self, start, end):
        assert len(start) == len(end), "Start and end coordinates must have the same length"
        # channel slicing
        return tuple([slice(None, None)] + [slice(start[i], end[i]) for i in range(len(start))])

    def _str_to_arr(self, data, key):
        if not isinstance(data[key], str):
            assert isinstance(
                data[key], (ListConfig, list, np.ndarray, torch.Tensor)
            ), "If coordinate arrays are not comma separated strings, they must be ListConfigs, lists, numpy arrays, or tensors"
            return data[key]
        return np.array([int(x) for x in data[key].split(",")])

    def __call__(self, data):
        if self.start_is_slice:
            slices = data[self.start_keys]
        elif self.crop_size is not None:
            # n_dims x n_crops
            start_coords = np.stack(
                [self._str_to_arr(data, start_key) for start_key in self.start_keys]
            )
            slices = [
                self._get_slice(start_coords[:, i], start_coords[:, i] + self.crop_size)
                for i in range(start_coords.shape[1])
            ]
        else:
            start_coords = np.stack(
                [self._str_to_arr(data, start_key) for start_key in self.start_keys]
            )
            end_coords = np.stack(
                [self._str_to_arr(data, start_key) for start_key in self.end_keys]
            )
            if start_coords.shape != end_coords.shape:
                raise ValueError(
                    "If both start and end coordinates are provided they should have the same length."
                )
            slices = [
                self._get_slice(start_coords[:, i], end_coords[:, i])
                for i in range(start_coords.shape[1])
            ]

        data_spatial_dims = {len(data[key].shape) for key in self.keys}
        if len(data_spatial_dims) != 1:
            raise ValueError(
                "All images should have the same spatial dimensions when applying center crops"
            )
        data_spatial_dims = data_spatial_dims.pop()
        assert np.all(
            [len(s) == data_spatial_dims for s in slices]
        ), "Slices should have the same dimensionality as the images"

        crops = [{key: data[key][s] for key in self.keys} for s in slices]
        return crops
