import math

import numpy as np
import torch
from monai.networks.blocks import Convolution, UnetOutBlock, UnetResBlock, UpSample

from aics_im2im.models.im2im.utils.postprocessing import detach

from .base_head import BaseHead


class ResBlocksHead(BaseHead):
    def __init__(
        self,
        loss,
        in_channels,
        out_channels,
        final_act,
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
