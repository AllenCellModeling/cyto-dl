from typing import Sequence

import torch
from escnn import gspaces, nn
from monai.networks.layers.convutils import (
    calculate_out_shape,
    same_padding,
    stride_minus_kernel_padding,
)


class SO2ImageEncoder(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        out_dim: int,
        channels: Sequence[int],
        strides: Sequence[int],
        maximum_frequency: int,
        kernel_size: int = 3,
        bias: bool = True,
        padding_mode: str = "replicate",
        relevance: bool = True,
    ):
        super().__init__()

        self.spatial_dims = spatial_dims
        self.kernel_size = kernel_size
        self.padding = same_padding(self.kernel_size)
        self.padding_mode = padding_mode
        self.bias = bias
        self.relevance = relevance

        self.gspace = (
            gspaces.rot2dOnR2(N=-1, maximum_frequency=maximum_frequency)
            if spatial_dims == 2
            else gspaces.rot2dOnR3(n=-1, maximum_frequency=maximum_frequency)
        )
        self.in_type = nn.FieldType(self.gspace, [self.gspace.trivial_repr])
        self.input_type = self.in_type

        blocks = [self.make_block(self.in_type, channels[0], strides[0])]
        for c, s in zip(channels[1:], strides[1:]):
            blocks.append(self.make_block(blocks[-1].out_type, c, s))

        fiber_gspace = gspaces.no_base_space(self.gspace.fibergroup)
        fiber_group = self.gspace.fibergroup
        pre_pooled_type = nn.FieldType(
            fiber_gspace,
            (
                [fiber_group.trivial_representation] * channels[-1]
                + [fiber_group.irrep(1)] * channels[-1]
            ),
        )
        embedding_type = nn.FieldType(fiber_gspace, [fiber_group.trivial_representation] * out_dim)
        pose_type = nn.FieldType(fiber_gspace, [fiber_group.irrep(1)])

        self.backbone = nn.SequentialModule(*blocks)

        norm_pool = nn.NormPool(pre_pooled_type)
        self.embedding_head = nn.SequentialModule(
            norm_pool, nn.Linear(norm_pool.out_type, embedding_type)
        )

        if self.relevance:
            self.relevance_head = self.make_conv(
                blocks[-1].out_type,
                nn.FieldType(self.gspace, [self.gspace.trivial_repr]),
                s=1,
                k=1,
                p=0,
                b=False,
            )
        else:
            self.relevance_head = None

        self.pose_head = nn.Linear(pre_pooled_type, pose_type)

    def forward(self, x):
        x = nn.GeometricTensor(x, self.input_type)
        y = self.backbone(x)

        sum_dims = (2, 3) if self.spatial_dims == 2 else (2, 3, 4)
        if self.relevance:
            relevance = self.relevance_head(y).tensor.sigmoid()
            y = y.tensor
            y = (y * relevance).sum(dim=sum_dims) / relevance.sum(dim=sum_dims)
        else:
            y = y.tensor
            y = y.mean(dim=sum_dims)

        y = self.embedding_head.in_type(y)
        y_embedding = self.embedding_head(y)
        y_pose = self.pose_head(y)

        return y_embedding, y_pose

    def get_fields(self, n_channels):
        scalar_fields = nn.FieldType(self.gspace, n_channels * [self.gspace.trivial_repr])
        vector_fields = nn.FieldType(self.gspace, n_channels * [self.gspace.irrep(1)])
        field_type = scalar_fields + vector_fields
        return scalar_fields, vector_fields, field_type

    def make_conv(self, in_type, out_type, s, k=None, p=None, b=None):
        k = k if k is not None else self.kernel_size
        p = p if p is not None else self.padding
        b = b if b is not None else self.bias

        conv_class = nn.R2Conv if self.spatial_dims == 2 else nn.R3Conv
        return conv_class(
            in_type,
            out_type,
            kernel_size=k,
            padding=p,
            stride=s,
            bias=b,
            padding_mode=self.padding_mode,
        )

    def make_block(self, in_type, out_channels, stride):
        out_scalar_fields, out_vector_fields, out_type = self.get_fields(out_channels)
        batch_norm_cls = nn.IIDBatchNorm2d if self.spatial_dims == 3 else nn.IIDBatchNorm3d

        return nn.SequentialModule(
            self.make_conv(in_type, out_type, stride),
            get_non_linearity(out_scalar_fields, out_vector_fields),
            get_batch_norm(out_scalar_fields, out_vector_fields, batch_norm_cls),
        )


def get_non_linearity(scalar_fields, vector_fields):
    out_type = scalar_fields + vector_fields
    relu = nn.ReLU(scalar_fields)
    norm_relu = nn.NormNonLinearity(vector_fields)
    nonlinearity = nn.MultipleModule(
        out_type,
        ["relu"] * len(scalar_fields) + ["norm"] * len(vector_fields),
        [(relu, "relu"), (norm_relu, "norm")],
    )
    return nonlinearity


def get_batch_norm(scalar_fields, vector_fields, batch_norm_cls):
    out_type = scalar_fields + vector_fields
    batch_norm = batch_norm_cls(scalar_fields)
    norm_batch_norm = nn.NormBatchNorm(vector_fields)
    batch_norm = nn.MultipleModule(
        out_type,
        ["bn"] * len(scalar_fields) + ["nbn"] * len(vector_fields),
        [(batch_norm, "bn"), (norm_batch_norm, "nbn")],
    )
    return batch_norm
