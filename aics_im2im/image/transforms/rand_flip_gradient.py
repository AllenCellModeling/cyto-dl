from typing import Optional, Sequence, Union

import torch
from monai.data.meta_obj import get_track_meta
from monai.transforms import RandFlip
from monai.utils import convert_to_tensor


class RandFlipGrad(RandFlip):
    """Transform to flip gradients."""

    def __init__(
        self,
        prob: float = 0.1,
        spatial_axis: Optional[Union[Sequence[int], int]] = None,
    ):
        """
        Parameters
        ----------
        prob:
            Probability of flipping.
        spatial_axis:
            Spatial axes along which to flip over. Default is None.
        """
        super().__init__(prob, spatial_axis)
        self.spatial_axis = spatial_axis

    def __call__(self, img: torch.Tensor, randomize: bool = True) -> torch.Tensor:
        """
        Parameters
        ----------
        img
            channel first array, must have shape: (num_channels, H[, W, ..., ]),
        randomize
            whether to execute `randomize()` function first, default to True.
        """
        if randomize:
            self.randomize(None)
        out = img
        if self._do_transform:
            out = self.flipper(img)
            # flip gradients of spatial_axis
            out[self.spatial_axis] *= -1
        out = convert_to_tensor(out, track_meta=get_track_meta())
        if get_track_meta():
            xform_info = self.pop_transform(out, check=False) if self._do_transform else {}
            self.push_transform(out, extra_info=xform_info)
        return out
