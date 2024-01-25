from typing import Dict, Optional, Union

import numpy as np
import torch
from monai.transforms import Transform
from monai.transforms.croppad.functional import pad_nd
from omegaconf import ListConfig, OmegaConf


class PadZd(Transform):
    """Transform for randomly padding top or bottom of crop by repeating first/last slice.

    Only applied if no segmentation is present in first/last slice
    """

    def __init__(
        self,
        image_key: str,
        segmentation_key: str,
        pad_amount: Dict[str, int],
        pad_keys: Union[str, ListConfig] = [],
        segmentation_ch: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        image_key: str
            name of images to pad
        segmentation_key: str
            name of segmentation. Used for checking whether top or bottom can be offset
        pad_amount: int
            number of slices to pad
        segmentation_ch: int
            channel of segmentation to check for presence of segmentation
        allow_missing_keys: bool
            allow missing keys
        """
        super().__init__()
        self.image_key = image_key
        self.segmentation_key = segmentation_key
        self.pad_keys = [image_key, segmentation_key] + pad_keys
        self.pad_amount = OmegaConf.to_container(pad_amount)
        self.segmentation_ch = segmentation_ch

    def __call__(self, img_dict):
        image = img_dict[self.image_key]
        segmentation = img_dict[self.segmentation_key]

        if segmentation.shape[0] > 1 and self.segmentation_ch is None:
            raise ValueError(
                "segmentation_ch must be specified if segmentation has more than one channel"
            )
        elif segmentation.shape[0] == 1:
            ch_seg = segmentation[0]
        else:
            ch_seg = segmentation[self.segmentation_ch]

        pad_mode = "replicate" if isinstance(image, torch.Tensor) else "edge"
        for key in self.pad_keys:
            pad = [(0, 0)] * 4  # high/low CZYX
            if (ch_seg[0] == 0).all():
                pad[1] = (self.pad_amount[key], 0)
            if (ch_seg[-1] == 0).all():
                # add upper padding
                pad[1] = np.max([pad[1], (0, self.pad_amount[key])], axis=1)
            img_dict[key] = pad_nd(img_dict[key], pad, mode=pad_mode)
        return img_dict
