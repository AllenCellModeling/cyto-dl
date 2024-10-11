import numpy as np
import torch
import torch.nn as nn


class ConvPoolReLU(torch.nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size):
        super().__init__()
        self.model = torch.nn.Sequential(
            nn.Conv3d(
                in_filters,
                out_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.model(x)


class STN(torch.nn.Module):
    def __init__(self, n_input_ch=2, patch_shape=(64, 256, 512), n_conv_filters=32):
        super().__init__()
        self.n_input_ch = n_input_ch
        depth = min(
            np.floor(
                np.log2(
                    np.min(patch_shape),
                )
            )
            - 1,
            4,
        )
        # final_number_of_convs * final_conv_output shape
        self.output_shape = 2 ** (depth - 1) * n_conv_filters * np.prod(patch_shape) // 8**depth

        in_filters = n_input_ch
        kernels = [7, 5, 5, 5, 3, 3, 3, 3, 3]
        localization = []
        for i in range(depth):
            out_filters = n_conv_filters * (2**i)
            localization.append(ConvPoolReLU(in_filters, out_filters, kernel_size=kernels[i]))
            in_filters = out_filters
        self.localization = nn.Sequential(*localization)

        self.fc_loc = nn.Sequential(
            nn.Linear(self.output_shape, 8 * n_conv_filters),
            nn.ReLU(True),
            nn.Linear(8 * n_conv_filters, 3),  # only output z, y, x
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x).view(-1, self.output_shape)
        offsets = self.fc_loc(xs).squeeze()
        # create identity transformation matrix with only shifts
        theta = torch.eye(3, 4).reshape(1, 3, 4).repeat(x.shape[0], 1, 1)
        theta[:, :, -1] = offsets

        # only align predicted channels
        x = x[:, : self.n_input_ch // 2]

        out_size = list(x.size())
        grid = nn.functional.affine_grid(theta, out_size)
        return (
            nn.functional.grid_sample(x, grid.type_as(x), padding_mode="border"),
            offsets,
        )

    def toggle(self, direction):
        assert isinstance(direction, bool)
        for p in self.parameters():
            p.requires_grad = direction
