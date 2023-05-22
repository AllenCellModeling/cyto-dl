from typing import Sequence

import torch
from escnn import gspaces, nn
from monai.networks.layers.convutils import (
    calculate_out_shape,
    same_padding,
    stride_minus_kernel_padding,
)


class ImageEncoder(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        out_dim: int,
        channels: Sequence[int],
        strides: Sequence[int],
        maximum_frequency: int,
        kernel_size: int = 3,
        bias: bool = False,
        group: str = "so2",
    ):
        super().__init__()

        self.spatial_dims = spatial_dims
        self.kernel_size = kernel_size
        self.padding = same_padding(self.kernel_size)
        self.bias = bias
        self.out_dim = out_dim
        self.group = group

        if group not in ("so2", "so3", None):
            raise ValueError(f"`gspace` should be one of ('so2', 'so3', None). Got {group}")

        if group == "so2":
            self.gspace = (
                gspaces.rot2dOnR2(N=-1, maximum_frequency=maximum_frequency)
                if spatial_dims == 2
                else gspaces.rot2dOnR3(n=-1, maximum_frequency=maximum_frequency)
            )
        elif group == "so3":
            if spatial_dims != 3:
                raise ValueError("The SO3 group only works for spatial_dims=3")
            self.gspace = gspaces.rot3dOnR3(maximum_frequency=maximum_frequency)
        else:
            self.gspace = gspaces.trivialOnR2() if spatial_dims == 2 else gspaces.trivialOnR3()

        self.in_type = nn.FieldType(self.gspace, [self.gspace.trivial_repr])

        out_vector_channels = None if group in ("so2", "so3") else 0

        blocks = [
            self.make_block(
                self.in_type, channels[0], strides[0], out_vector_channels=out_vector_channels
            )
        ]
        for c, s in zip(channels[1:], strides[1:]):
            blocks.append(
                self.make_block(blocks[-1].out_type, c, s, out_vector_channels=out_vector_channels)
            )

        if group == "so2":
            n_out_vectors = 1
        elif group == "so3":
            n_out_vectors = 2
        else:
            n_out_vectors = 0

        _, _, out_type = self.get_fields(out_dim, n_out_vectors)

        blocks.append(self.make_conv(blocks[-1].out_type, out_type, s=1, k=1, p=0, b=False))

        self.net = nn.SequentialModule(*blocks)

    def forward(self, x):
        x = nn.GeometricTensor(x, self.in_type)
        y = self.net(x)

        pool_dims = (2, 3) if self.spatial_dims == 2 else (2, 3, 4)
        y = y.tensor
        y = y.mean(dim=pool_dims)

        y_embedding = y[:, : self.out_dim]

        if self.group is not None:
            y_pose = y[:, self.out_dim :]

            if self.group == "so3":
                # separate two vectors into two channels
                y_pose = y_pose.reshape(y_pose.shape[0], 2, -1)

                # move from y z x (spharm) convention to x y z
                y_pose = y_pose[:, :, [2, 0, 1]]

            return y_embedding, y_pose
        return y_embedding

    def get_fields(self, n_channels, n_vector_channels=None):
        if n_vector_channels is None:
            n_vector_channels = n_channels

        scalar_fields = nn.FieldType(self.gspace, n_channels * [self.gspace.trivial_repr])

        if n_vector_channels > 0:
            vector_fields = nn.FieldType(self.gspace, n_vector_channels * [self.gspace.irrep(1)])
        else:
            vector_fields = []

        if n_vector_channels > 0:
            field_type = scalar_fields + vector_fields
        else:
            field_type = scalar_fields

        return scalar_fields, vector_fields, field_type

    def make_conv(self, in_type, out_type, s, k=None, p=None, b=None):
        k = k if k is not None else self.kernel_size
        p = p if p is not None else self.padding
        b = b if b is not None else self.bias

        conv_class = nn.R2Conv if self.spatial_dims == 2 else nn.R3Conv
        conv = conv_class(
            in_type,
            out_type,
            kernel_size=k,
            stride=s,
            bias=b,
            padding=(p if s == 1 else same_padding(k)),
        )

        return conv

    def make_block(self, in_type, out_channels, stride, out_vector_channels=None):
        out_scalar_fields, out_vector_fields, out_type = self.get_fields(
            out_channels, out_vector_channels
        )
        batch_norm_cls = nn.IIDBatchNorm3d if self.spatial_dims == 3 else nn.IIDBatchNorm2d

        return nn.SequentialModule(
            self.make_conv(in_type, out_type, stride),
            get_batch_norm(out_scalar_fields, out_vector_fields, batch_norm_cls),
            get_non_linearity(out_scalar_fields, out_vector_fields),
        )


def get_non_linearity(scalar_fields, vector_fields):
    nonlinearity = nn.ReLU(scalar_fields)
    if len(vector_fields) > 0:
        out_type = scalar_fields + vector_fields
        norm_relu = nn.NormNonLinearity(vector_fields)
        nonlinearity = nn.MultipleModule(
            out_type,
            ["relu"] * len(scalar_fields) + ["norm"] * len(vector_fields),
            [(nonlinearity, "relu"), (norm_relu, "norm")],
        )

    return nonlinearity


def get_batch_norm(scalar_fields, vector_fields, batch_norm_cls):
    batch_norm = batch_norm_cls(scalar_fields)
    if len(vector_fields) > 0:
        out_type = scalar_fields + vector_fields
        norm_batch_norm = nn.NormBatchNorm(vector_fields)
        batch_norm = nn.MultipleModule(
            out_type,
            ["bn"] * len(scalar_fields) + ["nbn"] * len(vector_fields),
            [(batch_norm, "bn"), (norm_batch_norm, "nbn")],
        )

    return batch_norm
