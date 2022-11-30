import torch
from monai.networks.blocks import SubpixelUpsample, Convolution


class AuxHead(torch.nn.Module):
    def __init__(
        self,
        resolution,
        in_channels,
        out_channels,
        final_act,
        n_hr_convs=1,
        dropout=0.1,
        hr_skip_channels=0,
    ):
        super().__init__()
        self.resolution = resolution
        modules = []
        conv_input_channels = in_channels
        if resolution == "hr":
            conv_input_channels //= 8
            self.upsample = SubpixelUpsample(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=conv_input_channels,
            )

            for i in range(n_hr_convs):
                # first convolution has to include skip connection from input
                modules.append(
                    Convolution(
                        spatial_dims=3,
                        in_channels=conv_input_channels + (hr_skip_channels) * (i == 0),
                        out_channels=conv_input_channels,
                        act=("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
                        norm="INSTANCE",
                        dropout=dropout,
                    )
                )
        modules.append(
            torch.nn.Conv3d(conv_input_channels, out_channels, kernel_size=1, padding=0)
        )
        modules.append(final_act)
        self.aux_head = torch.nn.Sequential(*modules)

    def forward(self, x, hr_skip):
        if self.resolution == "hr":
            x_hr = self.upsample(x)
            x = torch.cat((x_hr, hr_skip), dim=1)
        return self.aux_head(x)
