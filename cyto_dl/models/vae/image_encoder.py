from typing import Optional, Sequence, Tuple, Union

import torch
from escnn import gspaces, nn
from monai.networks.layers.convutils import (
    calculate_out_shape,
    same_padding,
    stride_minus_kernel_padding,
)
from omegaconf import ListConfig


class ImageEncoder(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        out_dim: int,
        channels: Sequence[int],
        strides: Sequence[int],
        maximum_frequency: int,
        kernel_sizes: Optional[Sequence[int]] = None,
        bias: bool = False,
        padding: Optional[Union[int, Sequence[int]]] = None,
        group: str = "so2",
        first_conv_padding_mode: str = "replicate",
        num_res_units: int = 2,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3] * len(channels)

        if padding is None:
            padding = [None] * len(channels)
        elif isinstance(padding, int):
            padding = [padding] * len(channels)
        else:
            assert isinstance(padding, (list, tuple, ListConfig))
            assert len(padding) == len(channels)

        self.spatial_dims = spatial_dims
        self.bias = bias
        self.out_dim = out_dim
        self.group = group
        self.num_res_units = num_res_units

        if group not in ("so2", "so3", None):
            raise ValueError(f"`gspace` should be one of ('so2', 'so3', None). Got {group!r}")

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

        blocks = [
            self.make_block(
                self.in_type,
                channels[0],
                strides[0],
                kernel_sizes[0],
                padding=padding[0],
                padding_mode=first_conv_padding_mode,
            )
        ]
        for c, s, k, p in zip(channels[1:], strides[1:], kernel_sizes[1:], padding[1:]):
            blocks.append(self.make_block(blocks[-1].out_type, c, s, k, padding=p))

        if group == "so2":
            n_out_vectors = 1
        elif group == "so3":
            n_out_vectors = 2
        else:
            n_out_vectors = 0

        blocks.append(
            self.make_block(
                blocks[-1].out_type,
                out_channels=out_dim,
                stride=1,
                kernel_size=1,
                padding=0,
                batch_norm=False,
                activation=False,
                out_vector_channels=n_out_vectors,
            )
        )

        self.net = nn.SequentialModule(*blocks)

    def make_block(
        self,
        in_type,
        out_channels,
        stride,
        kernel_size,
        padding,
        padding_mode="zeros",
        bias=True,
        batch_norm=True,
        activation=True,
        last_conv=False,
        out_vector_channels=None,
    ):
        if self.num_res_units > 0 and not last_conv:
            return ResBlock(
                spatial_dims=self.spatial_dims,
                in_type=in_type,
                out_channels=out_channels,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
                subunits=self.num_res_units,
                bias=bias,
            )

        return Convolution(
            spatial_dims=self.spatial_dims,
            in_type=in_type,
            out_channels=out_channels,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            bias=bias,
            batch_norm=batch_norm and not last_conv,
            activation=activation and not last_conv,
            out_vector_channels=out_vector_channels,
        )

    def forward(self, x):
        x = nn.GeometricTensor(x, self.in_type)
        y = self.net(x)

        pool_dims = (2, 3) if self.spatial_dims == 2 else (2, 3, 4)
        y = y.tensor.mean(dim=pool_dims)

        y_embedding = y[:, : self.out_dim]

        if self.group is not None:
            y_pose = y[:, self.out_dim :]

            if self.group == "so3":
                # separate two vectors into two channels
                y_pose = y_pose.reshape(y_pose.shape[0], 2, -1)  # TODO: check this

                # move from y z x (spharm) convention to x y z
                y_pose = y_pose[:, :, [2, 0, 1]]

            elif self.group == "so2":
                # move from y x (spharm) convention to x y
                y_pose = y_pose[:, [1, 0]]

            return y_embedding, y_pose
        return y_embedding


class ResBlock(nn.EquivariantModule):
    def __init__(
        self,
        spatial_dims,
        in_type,
        out_channels,
        stride,
        kernel_size,
        padding=None,
        padding_mode="zeros",
        subunits=2,
        bias=True,
    ):
        super().__init__()

        self.spatial_dims = spatial_dims
        self.in_type = in_type

        if padding is None:
            padding = same_padding(kernel_size)

        subunits = max(1, subunits)
        conv = []

        prev_out_type = in_type
        sstride = stride
        spadding = padding
        for su in range(subunits):
            unit = Convolution(
                spatial_dims=spatial_dims,
                in_type=prev_out_type,
                out_channels=out_channels,
                stride=sstride,
                kernel_size=kernel_size,
                bias=bias,
                padding=spadding,
                padding_mode=padding_mode,
                batch_norm=True,
                activation=True,
            )

            sstride = 1
            spadding = same_padding(kernel_size)
            conv.append(unit)
            prev_out_type = unit.out_type
        self.conv = nn.SequentialModule(*conv)

        need_res_conv = (
            stride != 1
            or in_type != self.conv.out_type
            or (stride == 1 and padding < same_padding(kernel_size))
        )

        if need_res_conv:
            rkernel_size = kernel_size
            rpadding = padding

            # if only adapting number of channels a 1x1 kernel is used with no padding
            if stride == 1 and padding == same_padding(kernel_size):
                rkernel_size = 1
                rpadding = 0

            self.residual = Convolution(
                spatial_dims=spatial_dims,
                in_type=in_type,
                out_channels=out_channels,
                stride=stride,
                kernel_size=rkernel_size,
                bias=bias,
                padding=rpadding,
                padding_mode=padding_mode,
                batch_norm=False,
                activation=False,
            )
        else:
            self.residual = nn.IdentityModule(in_type)
        self.out_type = self.conv.out_type

    def forward(self, x):
        return self.residual(x) + self.conv(x)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.in_type != self.out_type:
            return input_shape[:1] + (self.out_type.size,) + input_shape[2:]
        return input_shape


class Convolution(nn.EquivariantModule):
    def __init__(
        self,
        spatial_dims,
        in_type,
        out_channels,
        stride,
        kernel_size,
        bias=None,
        padding=None,
        padding_mode="zeros",
        batch_norm=True,
        activation=True,
        out_vector_channels=None,
    ):
        super().__init__()

        self.spatial_dims = spatial_dims
        self.in_type = in_type
        gspace = in_type.gspace
        group = in_type.gspace.fibergroup
        out_vector_channels = (
            out_vector_channels if out_vector_channels is not None else out_channels
        )

        scalar_fields = nn.FieldType(gspace, out_channels * [gspace.trivial_repr])
        if type(group).__name__ in ("SO2", "SO3"):
            vector_fields = nn.FieldType(gspace, out_vector_channels * [gspace.irrep(1)])
            out_type = scalar_fields + vector_fields
        else:
            vector_fields = []
            out_type = scalar_fields

        conv_class = nn.R3Conv if spatial_dims == 3 else nn.R2Conv
        conv = conv_class(
            in_type,
            out_type,
            kernel_size=kernel_size,
            stride=1,
            bias=bias,
            padding=padding,
            padding_mode=padding_mode,
        )

        if stride > 1:
            pool_class = (
                nn.PointwiseAvgPoolAntialiased2D
                if self.spatial_dims == 2
                else nn.PointwiseAvgPoolAntialiased3D
            )
            pool = pool_class(conv.out_type, sigma=0.33, stride=stride)
        else:
            pool = nn.IdentityModule(conv.out_type)

        if spatial_dims == 3 and batch_norm:
            batch_norm = get_batch_norm(scalar_fields, vector_fields, nn.IIDBatchNorm3d)
        elif spatial_dims == 2 and batch_norm:
            batch_norm = get_batch_norm(scalar_fields, vector_fields, nn.IIDBatchNorm2d)
        else:
            batch_norm = nn.IdentityModule(pool.out_type)

        if activation:
            activation = get_non_linearity(scalar_fields, vector_fields)
        else:
            activation = nn.IdentityModule(pool.out_type)

        self.net = nn.SequentialModule(conv, pool, batch_norm, activation)
        self.out_type = self.net.out_type

    def forward(self, x):
        return self.net(x)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.in_type != self.out_type:
            return input_shape[:1] + (self.out_type.size,) + input_shape[2:]
        return input_shape


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
