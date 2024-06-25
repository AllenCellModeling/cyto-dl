import numpy as np
import pandas as pd
import point_cloud_utils as pcu
import torch
from monai.transforms import Transform
from scipy import ndimage
from skimage import measure
from skimage.io import imread


def _rescale(img, thresh_max, thresh_min) -> torch.Tensor:
    if img.max() > thresh_max:
        img = torch.where(img > thresh_max, 0, img)
    if img.min() < thresh_min:
        img = torch.where(img > thresh_min, img, 0)
    return img


def _compute_labels(img, num_points, prob_exp):
    if (len(img.shape) > 3) & (img.shape[0] == 1):
        img = img[0]
    img = img - img.min()

    z, y, x = torch.where(torch.ones_like(img) > 0)
    probs = img.clone()
    probs_orig = img.clone()

    probs_orig = probs_orig.flatten()

    probs = probs.flatten()

    probs = probs / probs.max()
    probs = torch.exp(prob_exp * probs) - 1

    probs = probs / probs.sum()
    disp = 1

    idxs = torch.multinomial(probs, num_points, replacement=True).type_as(x)
    x = x[idxs] + 2 * (torch.rand(len(idxs)) - 0.5) * disp
    y = y[idxs] + 2 * (torch.rand(len(idxs)) - 0.5) * disp
    z = z[idxs] + 2 * (torch.rand(len(idxs)) - 0.5) * disp * 0.3
    probs = probs[idxs]
    probs_orig = probs_orig[idxs]
    new_cents = torch.stack([z, y, x, probs_orig], dim=1)

    return new_cents


class SamplePointCloud(Transform):
    def __init__(
        self,
        num_points: int,
        prob_exp: int,
        thresh_max: int,
        thresh_min: int,
    ):
        """Random rotate input on the XY plane. Assumes ZYX or ZXY ordering of coordinates if 3d.

        Parameters
        ----------
        spatial_dims: int
            Whether 2d or 3d
        """
        super().__init__()
        self.num_points = num_points
        self.prob_exp = prob_exp
        self.thresh_max = thresh_max
        self.thresh_min = thresh_min

    def __call__(self, img):
        img_res = _rescale(img, thresh_max=self.thresh_max, thresh_min=self.thresh_min)
        points = _compute_labels(img_res, self.num_points, self.prob_exp)
        return points


class SamplePointCloudd(Transform):
    def __init__(
        self,
        keys,
        num_points: int,
        prob_exp: int,
        thresh_max: int,
        thresh_min: int,
        out_key: int,
    ):
        """Dictionary-transform version of SO2RandomRotate."""
        super().__init__()
        self.keys = keys
        self.out_key = out_key
        self.transform = SamplePointCloud(num_points, prob_exp, thresh_max, thresh_min)

    def __call__(self, img):
        for key in self.keys:
            img[f"pc_{key}_{self.out_key}"] = self.transform(img[key])

        return img
