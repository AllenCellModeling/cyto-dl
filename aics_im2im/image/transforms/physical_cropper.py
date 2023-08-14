from typing import Dict, Sequence

import numpy as np
from monai.transforms import RandomizableTransform
from omegaconf import ListConfig


class RandomPhysicalDimsCropper(RandomizableTransform):
    """Monai-style transform that generates random slices at a given physical size."""

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
        pixel_key: str
            name of key that contains comma separated list  of physical pixel sizes (e.g. '[0.1,0.1,0.1]')
        physical_crop_size: Sequence[int]
            crop size (e.g. in microns)
        patch_per_image: int= 1
            Number of patches to sample per image
        """
        super().__init__()
        self.keys = keys if isinstance(keys, (list, ListConfig)) else [keys]
        self.physical_crop_size = np.asarray(physical_crop_size)
        self.pixel_key = pixel_key
        self.num_samples = patch_per_image
        self.spatial_dims = spatial_dims
        self.allow_missing_keys = allow_missing_keys

    @staticmethod
    def _generate_slice(start_coords: Sequence[int], roi_size: Sequence[int]) -> slice:
        """Creates slice starting at `start_coords` of size `roi_size`"""
        return tuple(
            [slice(None, None)]
            + [slice(start, end) for start, end in zip(start_coords, start_coords + roi_size)]
        )

    @staticmethod
    def _parse_pixel_str(pixel_str):
        try:
            pix = np.array(pixel_str[1:-1].split(","), dtype="float")
        except ValueError:
            raise ValueError('Check that your pixel sizes are of form "[1.0,2.0,3.0]"')
        return pix

    def generate_slices(self, image_dict: Dict, available_keys) -> Dict:
        """Generate dictionary of slices at all scales starting at random point."""
        pixel_dims = self._parse_pixel_str(image_dict[self.pixel_key])
        roi_size = (self.physical_crop_size / pixel_dims).astype(int)
        max_shape = image_dict[available_keys[0]].shape[-self.spatial_dims :]
        max_start_indices = max_shape - roi_size + 1
        if np.any(max_start_indices < 0):
            raise ValueError(f"Crop size {roi_size} is too large for image size {max_shape}")
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
            patch_dict = {key: image_dict[key][slices] for key in available_keys}
            patch_dict.update(meta_dict)
            patches.append(patch_dict)

        return patches
