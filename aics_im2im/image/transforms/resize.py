from typing import Sequence, Union

import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Transform


class Resized(Transform):
    """Transform to resize image by`scale_factor`"""

    def __init__(
        self,
        keys: Sequence[str],
        scale_factor: int,
        spatial_dims: int = 3,
        mode: str = "nearest",
        align_corners: Union[bool, None] = None,
        recompute_scale_factor: bool = False,
        antialias: bool = False,
        allow_missing_keys: bool = False,
    ):
        """
        Parameters
        ----------
        key: str
            name of images to resize
        scale_factor: int
            output size will be `img.shape*scale_factor`
        spatial_dims: int
            whether inputs are 2d or 3d
        mode:
            interpolation method. For more details see:
            https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html # noqa
        align_corners:
            see https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html # noqa
        recompute_scale_factor:
            see https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html # noqa
        antialias:
            see https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html # noqa
        """
        super().__init__()
        assert spatial_dims in (2, 3), f"Patch must be 2D or 3D, got {spatial_dims}"
        self.keys = keys
        self.scale_factor = np.asarray(scale_factor)
        self.spatial_dims = spatial_dims
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, img):
        for key in self.keys:
            if key in img.keys():
                out_size = list(
                    map(
                        round,
                        np.asarray(img[key].shape[-self.spatial_dims :]) * self.scale_factor,
                    )
                )
                raw_img = img[key]
                if len(raw_img.shape) != self.spatial_dims + 1:
                    raise ValueError("Images must have CZYX or CYX dimensions")
                raw_img = raw_img.as_tensor() if isinstance(raw_img, MetaTensor) else raw_img

                img[key] = torch.nn.functional.interpolate(
                    input=raw_img.unsqueeze(0),
                    size=out_size,
                    mode=self.mode,
                    align_corners=self.align_corners,
                    antialias=self.antialias,
                ).squeeze(0)
            elif not self.allow_missing_keys:
                raise KeyError(f"Key {key} not found in data. Available keys are {img.keys()}")
        return img
