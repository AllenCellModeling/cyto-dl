from typing import Callable, Dict, Sequence

import numpy as np
from monai.transforms import RandomizableTransform
from omegaconf import ListConfig


class RandomPhysicalDimsCropper(RandomizableTransform):
    """Monai-style transform that generates random slices at integer multiple scales.

    This can be useful for training superresolution models.
    """

    def __init__(
        self,
        keys: Sequence[str],
        pixel_key: str,
        physical_crop_size: Sequence[int],
        spatial_dims: int = 3,
        patch_per_image: int = 1,
        allow_missing_keys: bool = False,
    ):
        """
        Parameters
        ----------
        keys: Sequence[str]
            list of names dictionary keys to apply transform to
        x_key: str
            name of key that is passed into network. Its corresponding scale must be `1`
        physical_dims: Sequence[int]
            patch size to sample at resolution 1. Can have len 2 or 3
        patch_per_image: int= 1
            Number of patches to sample per image
        scales_dict: Dict
            Dictionary mapping scales key names to their resize factors.
            For example, `{raw:1, seg: [1.0, 0.5, 0.5]}` would take samples from `raw` of size
            `physical_dims` and samples from `seg` at `physical_dims`/[1.0,0.5, 0.5]
        selection_fn: Callable=None
            Function that takes in an image and returns True or False. Used to
            decide whether a sampled image should be kept or discarded.
        max_attempts: int=100
            max attempts to try sampling patches from an image before quitting
        """
        super().__init__()
        self.keys = keys if isinstance(keys, (list, ListConfig)) else [keys]
        self.physical_crop_size = np.asarray(physical_crop_size)
        self.pixel_key = pixel_key
        self.num_samples = patch_per_image
        self.spatial_dims = spatial_dims
        self.allow_missing_keys = allow_missing_keys

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
        return data[tuple(slicee)]

    @staticmethod
    def _generate_slice(start_coords: Sequence[int], roi_size: Sequence[int]) -> slice:
        """Creates slice starting at `start_coords` of size `roi_size`"""
        return [slice(None, None)] + [
            slice(start, end) for start, end in zip(start_coords, start_coords + roi_size)
        ]

    def generate_slices(self, image_dict: Dict, available_keys) -> Dict:
        """Generate dictionary of slices at all scales starting at random point."""
        roi_size = (self.physical_crop_size / np.asarray(image_dict[self.pixel_key])).astype(int)
        max_shape = image_dict[available_keys[0]].shape[-self.spatial_dims :]
        max_start_indices = max_shape - roi_size + 1
        if np.any(max_start_indices < 0):
            raise ValueError(f"Crop size {self.roi_size} is too large for image size {max_shape}")
        start_indices = self.R.randint(max_start_indices)
        return self._generate_slice(start_indices, roi_size)

    def __call__(self, image_dict):
        available_keys = self.keys
        if self.allow_missing_keys:
            available_keys = [k for k in self.keys if k in image_dict]

        meta_keys = set(image_dict.keys()) - set(available_keys)
        meta_dict = {mk: image_dict[mk] for mk in meta_keys}
        patches = []
        for _ in range(self.num_samples):
            slices = self.generate_slices(image_dict, available_keys)
            patch_dict = {
                key: self._apply_slice(image_dict[key], slices) for key in available_keys
            }
            patch_dict.update(meta_dict)
            patches.append(patch_dict)
        return patches
