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
        postprocess={"input": detach, "prediction": detach},
        model_args=None,
        calculate_metric=False,
        save_raw=False,
    ):
        super().__init__(loss, postprocess, model_args, calculate_metric, save_raw)

    def _init_model(self, model_args):
        resolution = model_args.get("resolution", "lr")
        self.resolution = resolution
        spatial_dims = model_args.get("spatial_dims", 3)
        n_convs = model_args.get("n_convs", 1)
        dropout = model_args.get("dropout", 0.0)
        out_channels = model_args["out_channels"]
        final_act = model_args["final_act"]
        in_channels = model_args["in_channels"]
        upsample_method = model_args.get("upsample_method", "subpixel")

        conv_input_channels = in_channels
        modules = [model_args.get("first_layer", torch.nn.Identity())]
        upsample = torch.nn.Identity()
        if resolution == "hr":
            if upsample_method == "subpixel":
                conv_input_channels //= 2**spatial_dims
            upsample_ratio = model_args.get("upsample_ratio", [2] * self.spatial_dims)
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
        model = torch.nn.ModuleDict({"upsample": upsample, "model": torch.nn.Sequential(*modules)})
        return model

    def forward(self, x):
        if self.resolution == "hr":
            x = self.model["upsample"](x)
        return self.model["model"](x)
