from typing import Optional

import numpy as np
import torch
from escnn.nn.modules.masking_module import build_mask
from monai.transforms import Transform


class RotationMask(Transform):
    def __init__(
        self,
        group: str,
        spatial_dims: int,
        mask_side: int,
        mask_margin: float = 0.0,
        background: Optional[int] = None,
        cylinder_axis: int = 1,  # CZYX
    ):
        """Similar functionality as https://quva-
        lab.github.io/escnn/api/escnn.nn.html#escnn.nn.MaskModule.

        In the 3d case, the mask is a cylinder - we only consider rotations of
        the XY plane.

        Parameters
        ----------
        group: str
            Rotation group (either so2 or so3)
        spatial_dims: int
            Whether 2d or 3d
        mask_side: int
            Side of the square in which to inscribe the mask.
            (Should be the max between the X and Y dimensions)
        mask_margin: float = 0.0
            Margin to smooth the mask edges. See the escnn docs linked above
            for details. Note: ignored if background is not `None`
        background: Optional[float] = None
            Value for masked out pixels/voxels. If `None` (default) background
            value will be 0. If not `None`, `mask_margin` is ignored and the
            mask is not smooth.
        cylinder_axis: int
            Axis along which to form the cylinder, in the 3d case.
            For rotations of the XY plane, this would be the Z axis.
            Defaults to 2, (i.e. assumes ZYX or ZXY ordering of spatial dimensions)
        """
        super().__init__()
        self.group = group[-2:].lower()
        assert self.group in ("o2", "o3")
        self.spatial_dims = spatial_dims
        self.background = background
        margin = mask_margin if background is None else 0

        if self.group == "o2":
            self.mask = build_mask(mask_side, dim=2, margin=margin).squeeze().unsqueeze(0)
            if self.spatial_dims == 3:
                self.mask = self.mask.unsqueeze(cylinder_axis)
        else:
            self.mask = build_mask(mask_side, dim=3, margin=margin).squeeze().unsqueeze(0)

    def __call__(self, img):
        if self.spatial_dims == 3 and len(img.shape) == 5:  # BCZYX
            mask = self.mask.unsqueeze(0)
        elif self.spatial_dims == 2 and len(img.shape) == 4:  # BCYX
            mask = self.mask.unsqueeze(0)
        else:
            mask = self.mask
        assert len(mask.shape) == len(img.shape)

        out = img * mask.type_as(img)
        if self.background is not None:
            out = out + self.background * (1 - mask.type_as(img))

        return out


class RotationMaskd(Transform):
    def __init__(
        self,
        keys,
        group: str,
        spatial_dims: int,
        mask_side: int,
        mask_margin: float = 0.0,
        background: Optional[float] = None,
        cylinder_axis: int = 1,  # CZYX
    ):
        """Dictionary-transform version of O2Mask."""
        super().__init__()
        self.keys = keys
        self.transform = RotationMask(
            group=group,
            spatial_dims=spatial_dims,
            mask_side=mask_side,
            mask_margin=mask_margin,
            background=background,
            cylinder_axis=cylinder_axis,
        )

    def __call__(self, img):
        for key in self.keys:
            img[key] = self.transform(img[key])

        return img
