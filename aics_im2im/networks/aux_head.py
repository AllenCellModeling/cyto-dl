import torch
from monai.networks.blocks import SubpixelUpsample, Convolution

class AuxHead(torch.nn.Module):
    def __init__(self,  resolution, in_channels, out_channels, final_act):
        super().__init__()
        modules = []
        conv_input_channels = in_channels
        if resolution == "hr":
            conv_input_channels //= 8
            upsample = SubpixelUpsample(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=conv_input_channels,
            )
            modules.append(upsample)
            modules.append(
                Convolution(
                    spatial_dims=3,
                    in_channels=conv_input_channels,
                    out_channels=conv_input_channels,
                    act=("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
                    norm="INSTANCE",
                    dropout=0.1,
                )
            )
        modules.append(
            torch.nn.Conv3d(
                conv_input_channels, out_channels, kernel_size=1, padding=0
            )
        )
        modules.append(final_act)
        self.aux_head = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.aux_head(x)
