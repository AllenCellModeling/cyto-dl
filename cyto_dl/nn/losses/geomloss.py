"""
Adapted from: https://github.com/jeanfeydy/geomloss
LICENSE: https://github.com/jeanfeydy/geomloss/blob/main/LICENSE.txt
"""

import torch
import torch.nn as nn
from geomloss import SamplesLoss


class GeomLoss(nn.Module):
    def __init__(
        self,
        name: str = "sinkhorn",
        p: int = 1,
        blur: float = 0.01,
        reach=None,
        scaling=0.5,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.p = p
        self.blur = blur
        self.reach = reach
        self.scaling = scaling
        kwargs.pop("_aux", None)

        self.loss = SamplesLoss(
            loss=self.name,
            p=self.p,
            blur=self.blur,
            reach=self.reach,
            scaling=self.scaling,
            **kwargs,
        )

    def forward(self, gts, preds):
        return self.loss(preds, gts)
