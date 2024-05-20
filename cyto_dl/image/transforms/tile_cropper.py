from typing import Dict, Sequence

import numpy as np
from monai.transforms import RandomizableTransform

class RandTileCropd(RandomizableTransform):
    """Monai-style transform that geenerates non-overlapping tiles starting at a random location
    """

    def __init__(
        self,
        keys: Sequence[str],
        patch_shape: Sequence[int],
        patch_per_image: int = 1,
        allow_missing_keys: bool = False,
    ):
        """
        Parameters
        ----------
        keys: Sequence[str]
            list of names dictionary keys to apply transform to
        patch_shape: Sequence[int]
            patch size to sample at resolution 1. Can have len 2 or 3
        patch_per_image: int= 1
            Number of patches to sample per image
        """
        super().__init__()
        assert len(patch_shape) in (
            2,
            3,
        ), f"Patch must be 2D or 3D, got {len(patch_shape)}"
        self.roi_size = np.asarray(patch_shape)
        self.keys = keys
        self.num_samples = patch_per_image
        self.spatial_dims = len(patch_shape)
        self.allow_missing_keys = allow_missing_keys

    def _coords_to_slice(self, start_coords: Sequence[int]) -> slice:
        """Creates slice starting at `start_coords` """
        return tuple([slice(None, None)] + [
            slice(start, end) for start, end in zip(start_coords, start_coords + self.roi_size)
        ])
    
    def _get_max_start_indices(self,  image: np.ndarray) -> np.ndarray:
        max_patches_per_dim = np.floor(np.array(image.shape[1:]) / self.roi_size).astype(int)
        if np.prod(max_patches_per_dim)< self.num_samples:
            raise ValueError(f"Image size {image.shape[1:]} is too small to generate {self.num_samples} patches of size {self.roi_size}, can only generate {max_patches_per_dim} patches in each dimension.")
        return image.shape[1:] - (self.roi_size * max_patches_per_dim), max_patches_per_dim


    def generate_slices(self, image: np.ndarray) -> Sequence[slice]:
        """Generate tiled slices starting at random point."""
        max_start_indices, max_patches_per_dim = self._get_max_start_indices(image)

        start_indices = self.R.randint(max_start_indices+1)

        available_crops = []
        for z in range(max_patches_per_dim[0]):
            for y in range(max_patches_per_dim[1]):
                for x in range(max_patches_per_dim[2]):
                    available_crops.append(self._coords_to_slice(
                        start_indices + np.array([z, y, x]) * self.roi_size
                    ))
        self.R.shuffle(available_crops)
        return available_crops[:self.num_samples]

    def __call__(self, image_dict):
        available_keys = self.keys
        if self.allow_missing_keys:
            available_keys = [k for k in self.keys if k in image_dict]

        meta_keys = set(image_dict.keys()) - set(available_keys)
        meta_dict = {mk: image_dict[mk] for mk in meta_keys}
        slices = self.generate_slices(image_dict[available_keys[0]])

        # Create patches for each slice
        patches = [
            {**{key: image_dict[key][slice_] for key in available_keys}, **meta_dict}
            for slice_ in slices
        ]

        return patches if len(patches) > 1 else patches[0]