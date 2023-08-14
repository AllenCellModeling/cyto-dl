from typing import List, Sequence, Union

import numpy as np
from monai.transforms import Transform
from omegaconf import ListConfig


class TokenCropd(Transform):
    """Monai-style transform that crops patches to multiples of a given size."""

    def __init__(
        self,
        keys: Sequence[str],
        spatial_dims: int = 3,
        n_patches: Union[int, Sequence[int]] = [4, 32, 32],
        allow_missing_keys: bool = False,
    ):
        """
        Parameters
        ----------
        keys: Sequence[str]
            list of names dictionary keys to apply transform to
        spatial_dims: int
            number of spatial dimensions
        n_patches: int
            number of patches to crop in each dimension
        allow_missing_keys: bool
            if True, allow missing keys
        """
        super().__init__()
        self.keys = keys if isinstance(keys, (list, ListConfig)) else [keys]
        self.spatial_dims = spatial_dims
        self.allow_missing_keys = allow_missing_keys
        n_patches = n_patches if isinstance(n_patches, (List, ListConfig)) else [n_patches]
        self.n_patches = np.asarray(n_patches)

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

            image_dict[key] = image_dict[key][:, : crop_size[0], : crop_size[1], : crop_size[2]]
        return image_dict
