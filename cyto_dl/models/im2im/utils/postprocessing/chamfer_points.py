from typing import Callable

import numpy as np
import torch
from hydra.utils import get_class
from numpy.typing import DTypeLike
from skimage.exposure import rescale_intensity
from skimage.measure import label


class ChamferPoints:
    """postprocessing transform for sampling points and applying chamfer loss"""

    def __init__(
        self,
        num_points: int = 4096,
        thresh_max: int = 0,
        thresh_min: int = 10,
        prob_exp: int = 20,
    ):
        """
        Parameters
        ----------
        num_points: Callable=torch.nn.Identity()
            activation to apply to image
        """
        self.num_points = num_points
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.prob_exp = prob_exp

    def _rescale(self, img: torch.Tensor) -> torch.Tensor:
        if img.max() > self.thresh_max:
            img = torch.where(img > self.thresh_max, 0, img)
        if img.min() < self.thresh_min:
            img = torch.where(img > self.thresh_min, img, 0)
        return img

    def _sample_points(self, img: torch.Tensor) -> torch.Tensor:

        img = self._rescale(img)

        img = img - img.min()

        if (len(img.shape) > 3) & (img.shape[0] == 1):
            img = img[0]

        z, y, x = torch.where(torch.ones_like(img) > 0)
        probs = img.clone()
        probs_orig = img.clone()
        
        probs_orig = probs_orig.flatten()
        
        probs = probs.flatten()
        
        probs = probs / probs.max()
        probs = torch.exp(self.prob_exp * probs) - 1
        
        probs = probs / probs.sum()
        disp = 1
        
        idxs = torch.multinomial(probs, self.num_points, replacement=True).type_as(x)

        x = x[idxs] + 2 * (torch.rand(len(idxs)).type_as(x) - 0.5) * disp
        y = y[idxs] + 2 * (torch.rand(len(idxs)).type_as(x) - 0.5) * disp
        z = z[idxs] + 2 * (torch.rand(len(idxs)).type_as(x) - 0.5) * disp * 0.3
        probs = probs[idxs]
        probs_orig = probs_orig[idxs]
        new_cents = torch.stack([z, y, x, probs], dim=1)
        return new_cents


    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        points = self._sample_points(img)
        return points
