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
from .vnn import VNLinear, VNLinearLeakyReLU, VNRotationMatrix

log = utils.get_pylogger(__name__)


def _make_conv(in_features, out_features, mode="scalar", scale_in=1, scale_out=1, final=False):
    in_features = in_features * scale_in
    out_features = out_features * scale_out

    if mode == "vector":
        return VNLinearLeakyReLU(
            in_features, out_features, use_batchnorm=False, negative_slope=0, dim=(4 if final else 5)
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
        hidden_features=[64, 64, 256, 512],
        k=20,
        mode="scalar",
        scalar_inds=None,
        include_cross=True,
        include_coords=True,
        get_rotation=False,
        break_symmetry=False,
    ):
        super().__init__()
        self.k = k
        self.num_features = num_features
        self.include_coords = include_coords
        self.include_cross = include_cross
        self.hidden_features = hidden_features
        self.scalar_inds = scalar_inds
        self.mode = mode

        self.break_symmetry = break_symmetry

        if self.break_symmetry:
            _features = [2] + hidden_features[:-1]
            if mode == "scalar":
                log.warn("Overriding `mode` to `vector` because symmetry breaking is on.")
                mode = "vector"
        else:
            _features = [1 if mode == "vector" else 3] + hidden_features[:-1]

        self.mode = mode
        convs = [_make_conv(_features[0], _features[1], mode, scale_in=1+include_coords+include_cross)]

        if self.mode ==  "scalar":
            num_channels=1+include_coords+include_cross
            start = 1
            final_input_features = sum(hidden_features[:-1])
            self.embedding = nn.Linear(hidden_features[-1], self.num_features, bias=False)
            self.pool = maxpool
        else:
            convs += [_make_conv(_features[1], _features[2], mode, scale_in=1, scale_out=2)]
            num_channels=2
            start = 2
            final_input_features = hidden_features[-1]*2
            self.embedding = VNLinear(hidden_features[-1], self.num_features)
            self.vn_inv = VNLinear(self.num_features, self.num_features)
            self.pool = meanpool
            self.rotation = VNRotationMatrix(self.num_features, dim=3, return_rotated=True)

        convs += [
            _make_conv(in_features, out_features, mode, scale_in=num_channels)
            for in_features, out_features in zip(_features[start:-1], _features[start+1:])
        ]
        self.convs = nn.ModuleList(convs)
            
        self.final_conv = _make_conv(
            final_input_features, hidden_features[-1], mode, scale_in=1, final=True
        )

            

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
        else:
            # per-element symmetry axis (e.g. moving direction)
            _axis = axis.view(x.shape[0], 1, 3, 1, 1)

        _axis = _axis.expand(x.shape[0], 1, 3, x.shape[-2], x.shape[-1])

        return torch.cat((x, _axis), dim=1)

    def forward(self, x, get_rotation=False, symmetry_breaking_axis=None):
        # x is [B, N, 3]
        x = x.transpose(2, 1)  # [B, 3, N]

        batch_size = x.size(0)

        intermediate_outs = []
        for idx, conv in enumerate(self.convs):
            if (idx == 0 and self.mode == 'vector') or (self.mode == 'scalar'):
                x = self.get_graph_features(x, idx)

            if idx == 0 and symmetry_breaking_axis is not None:
                if isinstance(symmetry_breaking_axis, int):
                    x = self.concat_axis(x, symmetry_breaking_axis)
                assert x.size(1) == 6
            pre_x = conv(x)

            if (len(pre_x.size()) < 5) and (self.mode == 'vector') and (idx > 0):
                if idx > 1:
                    x = self.pool(pre_x, dim=-1, keepdim=True).expand(pre_x.size())
                    x = torch.cat([x, pre_x], dim=1)
                else:
                    x = pre_x
            else:
                x = self.pool(pre_x, dim=-1)

            intermediate_outs.append(x)

        if self.mode == 'scalar':
            x = torch.cat(intermediate_outs, dim=1)
            x = self.final_conv(x)
            x = x.max(dim=-1, keepdim=False)[0]
        else:
            x = self.final_conv(x)
            x = self.pool(x, dim=-1)
        # import ipdb
        # ipdb.set_trace()
        embedding = self.embedding(x)

        if self.mode == 'vector':
            embedding, rot = self.rotation(embedding)
            embedding = self.vn_inv(embedding)
            rot = rot.mT
            embedding = embedding.norm(dim=-1)

        if get_rotation:
            return embedding, rot

        return embedding
