import numpy as np
import torch
from e3nn.o3 import ToS2Grid
from torch import Tensor
from torch.nn.modules.loss import _Loss as Loss

from cyto_dl.utils.spharm.rotation import get_band_indices


class SpharmLoss(Loss):
    def __init__(self, max_band, columns, res=1):
        super().__init__(None, None)

        assert res > 0
        assert res <= 1

        self.inverse = ToS2Grid(
            lmax=max_band, res=(int(34 * res), int(68 * res)), normalization="norm"
        )

        self.band_indices = get_band_indices(columns, max_band, flat=True)
        self._some_buffer = next(self.inverse.buffers())

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self._some_buffer.device != input.device:
            self.inverse = self.inverse.to(input.device)

        return (
            (
                self.inverse(input[:, self.band_indices])
                - self.inverse(target[:, self.band_indices])
            )
            .pow(2)
            .sum(dim=(1, 2))
        )
