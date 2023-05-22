from typing import Optional

import numpy as np
import torch
from monai.transforms import Transform


def make_mask(diam):
    xyz = np.indices((diam, diam, diam))

    rho = ((xyz - diam / 2) ** 2).sum(axis=0)

    return rho <= (diam / 2) ** 2


class O3Mask(Transform):
    def __init__(
        self,
        diameter: int,
        background: Optional[float] = None,
    ):
        """Similar functionality as https://quva-
        lab.github.io/escnn/api/escnn.nn.html#escnn.nn.MaskModule.

        Parameters
        ----------
        diameter: int
            Diameter of the ball to create
            (Should be the max between the Z, X and Y dimensions)
        background: float = 0.0
            Value for masked out pixels/voxels.
        """
        super().__init__()
        self.background = background
        self.mask = torch.tensor(make_mask(diameter))

    def __call__(self, img):
        if self.background is not None:
            return torch.where(
                self.mask.type_as(img) > 0, img, torch.tensor(self.background).type_as(img)
            )

        return img * self.mask.type_as(img)


class O3Maskd(Transform):
    def __init__(
        self,
        keys,
        diameter: int,
        background: Optional[float] = None,
    ):
        """Dictionary-transform version of O3Mask."""
        super().__init__()
        self.keys = keys
        self.transform = O3Mask(diameter, background)

    def __call__(self, img):
        for key in self.keys:
            img[key] = self.transform(img[key])

        return img
