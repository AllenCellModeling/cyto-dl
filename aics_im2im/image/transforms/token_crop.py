from typing import Sequence

import numpy as np
from monai.transforms import Transform
from omegaconf import ListConfig


class TokenCropd(Transform):
    """Monai-style transform that generates random slices at integer multiple scales.

    This can be useful for training superresolution models.
    """

    def __init__(
        self,
        keys: Sequence[str],
        spatial_dims: int = 3,
        n_patches: int = 12,
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
        self.spatial_dims = spatial_dims
        self.allow_missing_keys = allow_missing_keys
        self.n_patches = n_patches

    def __call__(self, image_dict):
        available_keys = self.keys
        if self.allow_missing_keys:
            available_keys = [k for k in self.keys if k in image_dict]

        for key in available_keys:
            patch_size = (
                np.asarray(image_dict[key].shape[-self.spatial_dims :]) / self.n_patches
            ).astype(int)
            assert np.all(
                np.asarray(patch_size) > 1
            ), f"All patch size must be >1 in all dimensions, got {patch_size}. Increase physical_crop_size or decrease number of patches. "
            crop_size = patch_size * self.n_patches

            # is this going to cause issues with resizing the patch embedding?
            image_dict[key] = image_dict[key][:, : crop_size[0], : crop_size[1], : crop_size[2]]

        return image_dict
