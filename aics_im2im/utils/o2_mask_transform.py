from typing import Optional

import numpy as np
import torch
from escnn.nn.modules.masking_module import build_mask
from monai.transforms import Transform


class O2Mask(Transform):
    def __init__(
        self,
        spatial_dims: int,
        mask_side: int,
        mask_margin: float = 0.0,
        background: Optional[int] = None,
        cylinder_axis: int = 2,
    ):
        """Similar functionality as https://quva-
        lab.github.io/escnn/api/escnn.nn.html#escnn.nn.MaskModule.

        In the 3d case, the mask is a cylinder - we only consider rotations of
        the XY plane.

        Parameters
        ----------
        spatial_dims: int
            Whether 2d or 3d
        mask_side: int
            Side of the square in which to inscribe the mask.
            (Should be the max between the X and Y dimensions)
        mask_margin: float = 0.0
            Margin to smooth the mask edges. See the escnn docs linked above
            for details. Note: ignored if background is not `None`
        background: Optional[int] = None
            Value for masked out pixels/voxels. If `None` (default) background
            value will be 0. If not `None`, `mask_margin` is ignored and the
            mask is not smooth.
        cylinder_axis: int
            Axis along which to form the cylinder, in the 3d case.
            For rotations of the XY plane, this would be the Z axis.
            Defaults to 2, (i.e. assumes ZYX or ZXY ordering of spatial dimensions)
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.background = background
        margin = mask_margin if background is None else 0

        self.mask = build_mask(mask_side, margin)
        if spatial_dims == 3:
            self.mask.unsqueeze(cylinder_axis)

    def __call__(self, img):
        if self.background is not None:
            return torch.where(self.mask, img, self.background)

        return img * self.mask.type_as(img)


class O2Maskd(Transform):
    def __init__(
        self,
        keys,
        spatial_dims: int,
        mask_side: int,
        mask_margin: float = 0.0,
        background: Optional[int] = None,
        cylinder_axis: int = 2,
    ):
        """Dictionary-transform version of O2Mask."""
        super().__init__()
        self.keys = keys
        self.transform = O2Mask(spatial_dims, mask_side, mask_margin, background, cylinder_axis)

    def __call__(self, img):
        for key in self.keys:
            img[key] = self.transform(img[key])

        return img
