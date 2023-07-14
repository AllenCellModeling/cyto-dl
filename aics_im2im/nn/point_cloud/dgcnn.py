"""Adapted from.

- https://github.com/FlyingGiraffe/vnn/blob/master/models/utils/vn_dgcnn_util.py
  LICENSE: https://github.com/FlyingGiraffe/vnn/blob/master/LICENSE
- https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/graph_functions.py
  LICENSE: https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/LICENSE_AnTao
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from aics_im2im import utils

from .graph_functions import get_graph_features
from .vnn import VNLeakyReLU, VNLinear, VNLinearLeakyReLU, VNRotationMatrix

log = utils.get_pylogger(__name__)


def _make_conv(
    in_features,
    out_features,
    mode="scalar",
    scale_in=1,
    include_symmetry=0,
    scale_out=1,
    final=False,
):
    in_features = in_features * scale_in + include_symmetry
    out_features = out_features * scale_out

    if mode == "vector":

        return VNLinearLeakyReLU(
            in_features,
            out_features,
            use_batchnorm=False,
            negative_slope=0,
            dim=(4 if final else 5),
        )

    conv = nn.Conv1d if final else nn.Conv2d
    batch_norm = nn.BatchNorm1d if final else nn.BatchNorm2d

    conv = conv(in_features, out_features, kernel_size=1, bias=False)
    # conv.weight.data.fill_(0.01)

    return nn.Sequential(
        conv,
        batch_norm(out_features),
        nn.LeakyReLU(negative_slope=0.2),
    )


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


class DGCNN(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_dim=64,
        k=20,
        mode="scalar",
        scalar_inds=None,
        include_cross=True,
        include_coords=True,
        get_rotation=False,
        symmetry_breaking_axis=None,
    ):
        super().__init__()
        self.k = k
        self.num_features = num_features
        self.include_coords = include_coords
        self.include_cross = include_cross
        self.hidden_dim = hidden_dim
        self.scalar_inds = scalar_inds
        self.mode = mode

        self.symmetry_breaking_axis = symmetry_breaking_axis
        include_symmetry = 0
        if self.symmetry_breaking_axis is not None:
            include_symmetry = 1
        print("include symmetry", include_symmetry)
        # if self.break_symmetry:
        #     _features = [2] + hidden_features[:-1]
        #     if mode == "scalar":
        #         log.warn("Overriding `mode` to `vector` because symmetry breaking is on.")
        #         mode = "vector"
        # else:
        #     _features = [1 if mode == "vector" else 3] + hidden_features[:-1]

        self.mode = mode
        self.init_features = 1 if self.mode == "vector" else 3
        convs = [
            _make_conv(
                self.init_features,
                self.hidden_dim,
                self.mode,
                scale_in=1 + include_coords + include_cross,
                include_symmetry=include_symmetry,
            )
        ]

        if self.mode == "scalar":
            convs += [_make_conv(self.hidden_dim * 2, self.hidden_dim)]
            convs += [_make_conv(self.hidden_dim * 2, self.hidden_dim * 2)]
            convs += [_make_conv(self.hidden_dim * 4, self.hidden_dim * 4)]

            self.final_len = self.hidden_dim * 2 + self.hidden_dim * 2 + self.hidden_dim * 4
            self.final_conv = _make_conv(self.final_len, self.final_len, final=True)
            self.embedding = nn.Linear(self.final_len, self.num_features, bias=False)
            self.pool = maxpool
        else:
            convs += [
                nn.Sequential(
                    VNLinear(self.hidden_dim, self.hidden_dim * 2),
                    VNLeakyReLU(2 * self.hidden_dim, negative_slope=0.0, share_nonlinearity=False),
                    VNLinear(2 * self.hidden_dim, self.hidden_dim),
                )
            ]
            for i in range(3):
                convs += [
                    nn.Sequential(
                        VNLeakyReLU(2 * hidden_dim, negative_slope=0.0, share_nonlinearity=False),
                        VNLinear(2 * hidden_dim, hidden_dim),
                    )
                ]

            self.final_conv = nn.Sequential(
                VNLeakyReLU(hidden_dim, negative_slope=0.0, share_nonlinearity=False),
                VNLinear(hidden_dim, self.num_features),
            )
            self.pool = meanpool
            self.rotation = VNRotationMatrix(self.num_features, dim=3, return_rotated=True)
            self.vn_inv = VNLinear(self.num_features, self.num_features)

        self.convs = nn.ModuleList(convs)
        self.embedding_head = nn.Linear(self.num_features, self.num_features)

    def get_graph_features(self, x, idx):
        return get_graph_features(
            x,
            k=self.k,
            mode=self.mode,
            scalar_inds=self.scalar_inds,
            include_cross=self.include_cross,
            # to make the scalar autoencoder equivariant we can refrain
            # from concatenating the input point coords to the output
            include_input=(self.mode == "vector" or self.include_coords or idx > 0),
        )

    def concat_axis(self, x, axis):
        if isinstance(axis, int):
            # symmetry axis is aligned with one of the X,Y,Z axes
            _axis = torch.zeros(3).float()
            _axis[axis] = 1.0
            _axis = _axis.view(1, 1, 3, 1, 1)
            # _axis = _axis.view(1, 3, 1, 1, 1)
        else:
            # per-element symmetry axis (e.g. moving direction)
            _axis = axis.view(x.shape[0], 1, 3, 1, 1)
            # _axis = axis.view(x.shape[0], 3, 1, 1, 1)

        _axis = _axis.expand(x.shape[0], 1, 3, x.shape[-2], x.shape[-1])
        # _axis = _axis.expand(x.shape[0], 3, 1, x.shape[-2], x.shape[-1])
        _axis = _axis.type_as(x)

        return torch.cat((x, _axis), dim=1)

    def forward(self, x, get_rotation=False):
        # x is [B, N, 3]
        x = x.transpose(2, 1)  # [B, 3, N]

        batch_size = x.size(0)

        intermediate_outs = []
        for idx, conv in enumerate(self.convs):
            if (idx == 0 and self.mode == "vector") or (self.mode == "scalar"):
                x = self.get_graph_features(x, idx)

            if idx == 0 and self.symmetry_breaking_axis is not None:
                if isinstance(self.symmetry_breaking_axis, int):
                    x = self.concat_axis(x, self.symmetry_breaking_axis)
                assert x.size(1) == 4
            pre_x = conv(x)

            if (len(pre_x.size()) < 5) and (self.mode == "vector") and (idx > 0):
                if (idx > 0) & (idx < len(self.convs) - 1):
                    x = self.pool(pre_x, dim=-1, keepdim=True).expand(pre_x.size())
                    x = torch.cat([x, pre_x], dim=1)
                else:
                    x = pre_x
            else:
                x = self.pool(pre_x, dim=-1)

            intermediate_outs.append(x)

        if self.mode == "scalar":
            x = torch.cat(intermediate_outs, dim=1)
            x = self.final_conv(x)
            x = x.max(dim=-1, keepdim=False)[0]
        else:
            x = self.final_conv(x)
            x = self.pool(x, dim=-1)
        # import ipdb
        # ipdb.set_trace()

        if self.mode == "vector":
            embedding, rot = self.rotation(x)
            embedding = self.vn_inv(embedding)
            rot = rot.mT
            embedding = torch.norm(embedding, dim=-1)
        else:
            embedding = self.embedding(x)

        embedding = self.embedding_head(embedding)

        if get_rotation:
            return embedding, rot

        return embedding
