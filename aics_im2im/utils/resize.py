from monai.transforms import Transform
import torch
import numpy as np


class Resized(Transform):
    def __init__(
        self,
        keys,
        scale_factor,
        mode="nearest",
        align_corners=None,
        recompute_scale_factor=False,
        antialias=False,
    ):
        super().__init__()
        self.keys = keys
        self.scale_factor = np.asarray(scale_factor)
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias

    def __call__(self, img):
        resized = {}
        for key in img.keys():
            if key in self.keys:
                out_size = list(
                    (np.asarray(img[key].shape[-3:]) * self.scale_factor).astype(int)
                )
                pre_shape = img[key].shape
                resized[key] = torch.nn.functional.interpolate(
                    input=img[key].unsqueeze(0).as_tensor(),
                    size=out_size,
                    mode=self.mode,
                    align_corners=self.align_corners,
                    antialias=self.antialias,
                ).squeeze(0)
            else:
                resized[key] = img[key]
        return resized
