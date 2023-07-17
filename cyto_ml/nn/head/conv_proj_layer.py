import math

import numpy as np
import torch
from monai.networks.blocks import Convolution


class ConvProjectionLayer(torch.nn.Module):
    """Layer for projecting e.g. 3D->2D image."""

    def __init__(self, dim, pool_size: int, in_channels: int, out_channels: int):
        """
        Parameters
        ---------
        dim
            Dimension to project, e.g. 2 for projecting NCZYX -> NCYX
        pool_size:int
            Size of convolutional kernel for downsampling
        in_channels:int
            number of input channels
        out_channels:int
            number of output channels
        """
        super().__init__()
        self.dim = dim
        n_downs = math.floor(np.log2(pool_size))
        modules = []
        for _ in range(n_downs):
            modules.append(
                Convolution(
                    spatial_dims=3,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=[2, 1, 1],
                    strides=[2, 1, 1],
                    padding=[0, 0, 0],
                )
            )
        remainder = pool_size - 2**n_downs
        if remainder != 0:
            modules.append(
                Convolution(
                    spatial_dims=3,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=[remainder, 1, 1],
                    strides=[remainder, 1, 1],
                    padding=[0, 0, 0],
                )
            )
        self.model = torch.nn.Sequential(*modules)

    def __call__(self, x):
        return self.model(x).squeeze(self.dim)
