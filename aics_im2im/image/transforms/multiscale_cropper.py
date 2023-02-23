from typing import Callable, Dict, Sequence

import numpy as np
from monai.transforms import RandomizableTransform


def _apply_slice(data, slicee):
    """If the slice is generated for 3d images, but the passed data is a 2d image (e.g. if we are
    predicting a 3d image and a 2d image), this will apply the CXY portions of the slice.

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
        return data[[x for i, x in enumerate(slicee) if i != 1]]
    return data[slicee]


def _generate_slice(start_coords: Sequence[int], roi_size: Sequence[int]) -> slice:
    """Creates slice starting at `start_coords` of size `roi_size`"""
    return [slice(None, None)] + [
        slice(start, end) for start, end in zip(start_coords, start_coords + roi_size)
    ]


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
    ):
        """
        Parameters
        ----------
        keys: Sequence[str]
            list of names dictionary keys to apply transform to
        patch_shape: Sequence[int]
            patch size to sample at resolution 1. Can be 2 or 3
        patch_per_image: int= 1
            Number of patches to sample per image
        scales_dict: Dict
            Dictionary mapping scales (e.g. 1, 2, ...n) to key names.
            For example, `{1: seg, 2: raw}` would take samples from `seg` at
            `patch_shape` and samples from `raw` at `patch_shape`/2
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
        self.scale_dict = scales_dict
        self.reversed_scale_dict = {}
        self.selection_fn = selection_fn
        self.max_attempts = max_attempts
        self.spatial_dims = len(patch_shape)

        # reversed scales dict is used to map from a key to scale for sampling
        for k, v in scales_dict.items():
            for v_item in v:
                self.reversed_scale_dict[v_item] = k
        assert 1 in self.scale_dict.keys()

    def generate_slices(self, image_dict: Dict) -> Dict:
        """Generate dictionary of slices at all scales starting at random point."""
        max_shape = np.asarray(image_dict[self.scale_dict[1][0]].shape[-self.spatial_dims :])
        max_start_indices = max_shape - self.roi_size + 1
        if np.any(max_start_indices < 0):
            raise ValueError(f"Crop size {self.roi_size} is too large for image size {max_shape}")
        start_indices = self.R.randint(max_start_indices)
        scaled_start_indices = {
            s: (start_indices // s).astype(int) for s in self.scale_dict.keys()
        }
        return {
            s: _generate_slice(scaled_start_indices[s], (self.roi_size // s).astype(int))
            for s in scaled_start_indices.keys()
        }

    def __call__(self, image_dict):
        patches = []
        attempts = 0
        while len(patches) < self.num_samples:
            if attempts > self.max_attempts:
                raise ValueError(
                    "Max attempts reached. Please check your selection function "
                    "or adjust max_attempts"
                )
            slices = self.generate_slices(image_dict)

            patch_dict = {
                key: _apply_slice(image_dict[key], slices[self.reversed_scale_dict[key]])
                for key in self.keys
            }

            if self.selection_fn is None or self.selection_fn(patch_dict):
                patches.append(patch_dict)
                attempts = 0

            attempts += 1

        return patches
