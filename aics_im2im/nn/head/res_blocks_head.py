import math
from typing import Callable

import numpy as np
import torch
from monai.networks.blocks import Convolution, UnetOutBlock, UnetResBlock, UpSample

from aics_im2im.models.im2im.utils.postprocessing import detach

from .base_head import BaseHead


class ResBlocksHead(BaseHead):
    """Task head for doing task-specific convolution and optional upsampling."""

    def __init__(
        self,
        loss,
        in_channels: int,
        out_channels: int,
        final_act: Callable = torch.nn.Identity(),
        postprocess={"input": detach, "prediction": detach},
        calculate_metric=False,
        save_raw=False,
        resolution="lr",
        spatial_dims=3,
        n_convs=1,
        dropout=0.0,
        upsample_method="subpixel",
        upsample_ratio=None,
        first_layer=torch.nn.Identity(),
    ):
        """
        Parameters
        ----------
        in_channels:int
            Number of input channels (same as number of output channels from backbone)
        out_channels:int
            Number of output channels
        final_act:Callable=torch.nn.Identity()
            Final activation applied to logits
        postprocess={"input": detach, "prediction": detach}
            Postprocessing functions for ground truth and model predictions
        calculate_metric=False
            Whether to calculate a metric. Currently not implemented
        save_raw=False
            Whether to save raw image examples during training
        resolution="lr"
            Resolution of output image. If `lr`, no upsampling is done. If `hr`, `upsample_method` and `upsample_ratio` are used
            to determine how to perform upsampling.
        spatial_dims=3
            Spatial dimension of data after `first_layer`
        n_convs=1
            Number of convolutional layers
        dropout=0.0
            Dropout ratio
        upsample_method="subpixel"
            Method of upsampling. See :ref:monai's docs`https://docs.monai.io/en/stable/networks.html#monai.networks.blocks.Upsample` for options
        upsample_ratio=None
            Amount to upsample. If not None, should be array of length `spatial_dims`
        first_layer=torch.nn.Identity()
            Initial layer to apply to backbone outputs. For example, `ConvProjectionLayer` for transforming 3D->2D output.
        """
        super().__init__(loss, postprocess, calculate_metric, save_raw)

        self.resolution = resolution
        conv_input_channels = in_channels
        modules = [first_layer]
        upsample = torch.nn.Identity()

        upsample_ratio = upsample_ratio or [2] * spatial_dims

        if resolution == "hr":
            if upsample_method == "subpixel":
                conv_input_channels //= 2**spatial_dims
            assert len(upsample_ratio) == spatial_dims
            upsample = UpSample(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=conv_input_channels,
                scale_factor=upsample_ratio,
                mode=upsample_method,
            )
        for _ in range(n_convs):
            in_channels = conv_input_channels
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
        self.model = torch.nn.ModuleDict(
            {"upsample": upsample, "model": torch.nn.Sequential(*modules)}
        )

    def forward(self, x):
        if self.resolution == "hr":
            x = self.model["upsample"](x)
        return self.model["model"](x)
