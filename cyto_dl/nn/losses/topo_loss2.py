"""
Adapted from: https://github.com/search?q=repo%3Amarrlab%2FSHAPR_torch+torch_top&type=code
SHAPR
"""

import torch
import torch.nn as nn
from torch_topological.nn import CubicalComplex, VietorisRipsComplex
from torch_topological.nn import SignatureLoss
from torch_topological.nn.data import batch_iter
from torch_topological.utils import total_persistence


class TopoLoss2(nn.Module):
    def __init__(
        self,
        dim: int = 1,
        p: int = 2,
        topo_lambda: float = 1,
        **kwargs,
    ):
        super().__init__()
        self.topo_lambda = topo_lambda
        self.dim = dim
        self.p = p
        self.complex = VietorisRipsComplex(dim=self.dim, p=self.p)
        # self.loss = WassersteinDistance()
        self.loss = SignatureLoss(p=2)

    def forward(self, gts, z):
        pi_x = self.complex(gts)
        pi_y = self.complex(z)

        topo_loss = self.loss([gts, pi_x], [z, pi_z])

        return tl * self.topo_lambda
