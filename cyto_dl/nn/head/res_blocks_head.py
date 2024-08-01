from typing import Callable

import torch
from monai.networks.blocks import DenseBlock, UnetOutBlock, UnetResBlock, UpSample

from cyto_dl.models.im2im.utils.postprocessing import detach

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
        resolution="lr",
        spatial_dims=3,
        n_convs=1,
        dropout=0.0,
        upsample_method="pixelshuffle",
        upsample_ratio=None,
        first_layer=torch.nn.Identity(),
        dense: bool = False,
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
        resolution="lr"
            Resolution of output image. If `lr`, no upsampling is done. If `hr`, `upsample_method` and `upsample_ratio` are used
            to determine how to perform upsampling.
        spatial_dims=3
            Spatial dimension of data after `first_layer`
        n_convs=1
            Number of convolutional layers
        dropout=0.0
            Dropout ratio
        upsample_method="pixelshuffle"
            Method of upsampling. See the [monai upsampling docs](https://docs.monai.io/en/stable/networks.html#monai.networks.blocks.Upsample) for options
        upsample_ratio=None
            Amount to upsample. If not None, should be array of length `spatial_dims`
        first_layer=torch.nn.Identity()
            Initial layer to apply to backbone outputs. For example, `ConvProjectionLayer` for transforming 3D->2D output.
        dense=False
            Whether to use dense connections between convolutional layers
        """
        super().__init__(loss, postprocess)

        self.resolution = resolution
        conv_input_channels = in_channels
        modules = [first_layer]
        upsample = torch.nn.Identity()

        if isinstance(upsample_ratio, int):
            upsample_ratio = [upsample_ratio] * spatial_dims

        if resolution == "hr":
            if upsample_method == "pixelshuffle":
                conv_input_channels //= 2**spatial_dims
            assert len(upsample_ratio) == spatial_dims
            upsample = UpSample(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=conv_input_channels,
                scale_factor=upsample_ratio,
                mode=upsample_method,
            )
        for i in range(n_convs):
            in_channels = conv_input_channels
            if dense:
                in_channels = (i + 1) * conv_input_channels
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
        if dense:
            # dense convolutions
            modules = [modules[0]] + [DenseBlock(modules[1:])]
            conv_input_channels *= n_convs + 1
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
