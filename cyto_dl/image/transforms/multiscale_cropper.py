from typing import Callable, Dict, Sequence

import numpy as np
from monai.transforms import RandomizableTransform
from omegaconf import ListConfig


class RandomMultiScaleCropd(RandomizableTransform):
    """Monai-style transform that generates random slices at integer multiple scales.

    This can be useful for training superresolution models.
    """

    def __init__(
        self,
        keys: Sequence[str],
        patch_shape: Sequence[int],
        scales_dict: Dict,
        patch_per_image: int = 1,
        selection_fn: Callable = None,
        max_attempts: int = 100,
        allow_missing_keys: bool = False,
    ):
        """
        Parameters
        ----------
        keys: Sequence[str]
            list of names dictionary keys to apply transform to
        x_key: str
            name of key that is passed into network. Its corresponding scale must be `1`
        patch_shape: Sequence[int]
            patch size to sample at resolution 1. Can have len 2 or 3
        patch_per_image: int= 1
            Number of patches to sample per image
        scales_dict: Dict
            Dictionary mapping scales key names to their resize factors.
            For example, `{raw:1, seg: [1.0, 0.5, 0.5]}` would take samples from `raw` of size
            `patch_shape` and samples from `seg` at `patch_shape`/[1.0,0.5, 0.5]
        selection_fn: Callable=None
            Function that takes in an image and returns True or False. Used to
            decide whether a sampled image should be kept or discarded.
        max_attempts: int=100
            max attempts to try sampling patches from an image before quitting
        """
        super().__init__()
        assert len(patch_shape) in (
            2,
            3,
        ), f"Patch must be 2D or 3D, got {len(patch_shape)}"
        self.roi_size = np.asarray(patch_shape)
        self.keys = keys
        self.num_samples = patch_per_image
        self.reversed_scale_dict = {}
        self.selection_fn = selection_fn
        self.max_attempts = max_attempts
        self.spatial_dims = len(patch_shape)
        self.allow_missing_keys = allow_missing_keys

        self.scale_dict = {}
        for k, v in scales_dict.items():
            if k not in keys:
                continue
            if isinstance(v, (list, ListConfig)):
                assert len(v) in (
                    1,
                    self.spatial_dims,
                ), f"If list is passed to multiscale cropper, must have len 1 or {self.spatial_dims}, got {len(v)}"
                if len(v) == 1:
                    v = np.tile(v, self.spatial_dims)
            else:
                v = np.ones(self.spatial_dims) * v
            self.scale_dict[k] = np.asarray(v)

    @staticmethod
    def _apply_slice(data, slicee):
        """If the slice is generated for 3d images, but the passed data is a 2d image (e.g. if we
        are predicting a 3d image and a 2d image), this will apply the CXY portions of the slice.

        Parameters
        ----------
        data:
            image with shape C[Z]YX
        slice:
            slice object with length 3 (generated for 2d images) or 4
            (generated for 3d images)
        """
        # pop z dimension to take corresponding 2d slice if 2d image is passed
        if len(data.shape) == len(slicee) - 1:
            return data[tuple(x for i, x in enumerate(slicee) if i != 1)]
        return data[tuple(slicee)]

    @staticmethod
    def _generate_slice(start_coords: Sequence[int], roi_size: Sequence[int]) -> slice:
        """Creates slice starting at `start_coords` of size `roi_size`"""
        return [slice(None, None)] + [  # noqa: FURB140
            slice(start, end) for start, end in zip(start_coords, start_coords + roi_size)
        ]

    def _get_max_start_indices(self, image_dict: Dict):
        """Find crop start coordinates within bounds across all images/scales given roi size."""
        max_start_indices = np.ones(self.spatial_dims) * np.inf
        for im_name, rescale_factor in self.scale_dict.items():
            if im_name not in image_dict:
                continue
            shape = np.asarray(image_dict[im_name].shape[-self.spatial_dims :])
            shape = np.floor_divide(shape, rescale_factor)
            roi_size = np.floor_divide(self.roi_size, rescale_factor)
            max_start_indices_img = shape - roi_size
            max_start_indices = np.minimum(max_start_indices_img, max_start_indices)
            if np.any(max_start_indices < 0):
                raise ValueError(f"Crop size {roi_size} is too large for image size {shape}")
            max_start_indices += max_start_indices == 0
        return max_start_indices

    def generate_slices(self, image_dict: Dict) -> Dict:
        """Generate dictionary of slices at all scales starting at random point."""
        max_start_indices = self._get_max_start_indices(image_dict)

        start_indices = self.R.randint(max_start_indices)

        scaled_start_indices = {
            k: (start_indices * v).astype(int) for k, v in self.scale_dict.items()
        }

        return {
            k: self._generate_slice(scaled_start_indices[k], (self.roi_size * v).astype(int))
            for k, v in self.scale_dict.items()
        }

    def __call__(self, image_dict):
        available_keys = self.keys
        if self.allow_missing_keys:
            available_keys = [k for k in self.keys if k in image_dict]

        meta_keys = set(image_dict.keys()) - set(available_keys)
        meta_dict = {mk: image_dict[mk] for mk in meta_keys}
        patches = []
        attempts = 0
        while len(patches) < self.num_samples:
            if attempts > self.max_attempts:
                raise ValueError(
                    "Max attempts reached. Please check your selection function "
                    "or adjust max_attempts"
                )
            slices = self.generate_slices({k: image_dict[k] for k in available_keys})

            patch_dict = {
                key: self._apply_slice(image_dict[key], slices[key]) for key in available_keys
            }

            patch_dict.update(meta_dict)
            if self.selection_fn is None or self.selection_fn(patch_dict):
                patches.append(patch_dict)
                attempts = 0

            attempts += 1
        return patches if len(patches) > 1 else patches[0]
