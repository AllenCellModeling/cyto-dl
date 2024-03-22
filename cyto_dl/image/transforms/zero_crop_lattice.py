from typing import Sequence

import numpy as np
from monai.transforms import Transform


class ZeroCropLatticed(Transform):
    """Crop zeros from top, bottom, and diagonal of deskewed lattice images."""

    def __init__(
        self,
        keys: Sequence[str],
        spatial_dims: int = 3,
        allow_missing_keys: bool = False,
    ):
        super().__init__()
        self.keys = keys
        self.spatial_dims = spatial_dims
        self.lattice_crop = ZeroCropLattice(spatial_dims)

    def __call__(self, img_dict):
        # check if self.keys exist in img_dict
        for key in self.keys:
            if key not in img_dict and not self.allow_missing_keys:
                raise KeyError(f"img_dict must contain key: {key}")
            elif key in img_dict:
                img_dict[key] = self.lattice_crop(img_dict[key])

        return img_dict


class ZeroCropLattice(Transform):
    """Crop zeros from top, bottom, and diagonal of deskewed lattice images."""

    def __init__(
        self,
        spatial_dims: int = 3,
    ):
        self.spatial_dims = spatial_dims

    def __call__(self, img):
        img = img.squeeze()
        assert len(img.shape) == 3
        z, _, _ = np.where(img > 0)
        z_min, z_max = np.min(z), np.max(z)
        min_project = np.min(img[z_min:z_max], axis=0)
        y, x = np.where(min_project > 0)
        y_min, y_max, x_min, x_max = np.min(y), np.max(y), np.min(x), np.max(x)
        cropped = img[z_min:z_max, y_min:y_max, x_min:x_max]
        return cropped.unsqueeze(0)
