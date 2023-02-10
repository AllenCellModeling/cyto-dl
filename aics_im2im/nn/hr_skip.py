import torch
from monai.networks.blocks import Convolution


class HRSkip(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0.3,
        scale_factor=[2, 2, 2],
        mode="nearest",
        align_corners=None,
        antialias=False,
    ):
        super().__init__()
        self.model = Convolution(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            act=("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
            norm="INSTANCE",
            dropout=dropout,
        )
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.antialias = antialias

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            input=x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            antialias=self.antialias,
        )
        return self.model(x)
