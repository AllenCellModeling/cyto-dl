import math

import numpy as np
import torch
from monai.networks.blocks import (
    Convolution,
    SubpixelUpsample,
    UnetOutBlock,
    UnetResBlock,
)


class IdentityAuxHead(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, hr_skip):
        return x


class PoolProjectionLayer(torch.nn.Module):
    def __init__(self, dim, pool_size):
        super().__init__()
        self.dim = dim
        self.projection = torch.nn.MaxPool3d(kernel_size=[pool_size, 1, 1])

    def __call__(self, x):
        return self.projection(x).squeeze(self.dim)


class ConvProjectionLayer(torch.nn.Module):
    def __init__(self, dim, pool_size, in_channels, out_channels):
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


class AuxHead(torch.nn.Module):
    def __init__(
        self,
        resolution,
        in_channels,
        out_channels,
        final_act,
        n_convs=1,
        dropout=0.1,
        hr_skip_channels=0,
        spatial_dims=3,
        first_layer=torch.nn.Identity(),
    ):
        super().__init__()
        self.resolution = resolution
        conv_input_channels = in_channels
        modules = [first_layer]
        if resolution == "hr":
            conv_input_channels //= 2**spatial_dims
            self.upsample = SubpixelUpsample(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=conv_input_channels,
            )

        for i in range(n_convs):
            in_channels = conv_input_channels
            # first hr block
            if i == 0 and resolution == "hr":
                in_channels += hr_skip_channels

            modules.append(
                UnetResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=conv_input_channels,
                    stride=1,
                    kernel_size=3,
                    norm_name="INSTANCE",
                    dropout=dropout,
                )
            )
        modules.extend(
            (
                UnetOutBlock(
                    spatial_dims=spatial_dims,
                    in_channels=conv_input_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                ),
                final_act,
            )
        )
        self.aux_head = torch.nn.Sequential(*modules)

    def forward(self, x, hr_skip):
        if self.resolution == "hr":
            x_hr = self.upsample(x)
            x = torch.cat((x_hr, hr_skip), dim=1)
        return self.aux_head(x)
