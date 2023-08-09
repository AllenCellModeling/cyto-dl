"""
Adapted from: https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/chamfer_distance.py
LICENSE: https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/LICENSE_AnTao
"""

import torch
import torch.nn as nn
from geomloss import SamplesLoss


class GeomLoss(nn.Module):
    def __init__(
        self, name: str = "sinkhorn", p: int = 1, blur: float = 0.01, **kwargs
    ):
        super().__init__()
        self.name = name
        self.p = p
        self.blur = blur
        self.loss = SamplesLoss(loss=self.name, p=self.p, blur=self.blur, **kwargs)

    def forward(self, gts, preds):
        return self.loss(preds, gts)
