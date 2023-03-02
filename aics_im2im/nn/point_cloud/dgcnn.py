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


def _make_conv(in_features, out_features, mode="scalar"):
    if mode == "vector":
        VNLinearLeakyReLU(in_features, out_features, use_batchnorm=True)

    return nn.Sequential(
        nn.Conv2d(in_features * 2, out_features, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_features),
        nn.LeakyReLU(negative_slope=0.2),
    )


class DGCNN(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_features=[64, 128, 256, 512],
        k=20,
        mode="scalar",
        include_cross=True,
        include_coords=True,
        get_rotation=False,
        break_symmetry_axis=None,
    ):
        super().__init__()
        self.k = k
        self.num_features = num_features
        self.include_coords = include_coords
        self.include_cross = include_cross

        self.break_symmetry_axis = break_symmetry_axis

        if self.break_symmetry_axis:
            _features = [6] + hidden_features[:-1]
            if mode == "scalar":
                log.warn("Overriding `mode` to `vector` because symmetry breaking is on.")
                mode = "vector"
        else:
            _features = [3] + hidden_features[:-1]

        self.mode = mode

        convs = [
            _make_conv(in_features, out_features, mode)
            for in_features, out_features in zip(_features[:-1], _features[1:])
        ]
        convs = nn.ModuleList(convs)

        final_input_features = np.prod(hidden_features[:-1])
        self.final_conv = _make_conv(final_input_features, hidden_features[-1], mode)

        if mode == "scalar":
            self.embedding = nn.Linear(hidden_features[-1], self.num_features, bias=False)
        else:
            self.embedding = VNLinear(hidden_features[-1], self.num_features)

        if mode == "vector" or get_rotation:
            self.rotation = VNRotationMatrix(hidden_features[-1])

    def get_graph_features(self, x, idx):
        return get_graph_features(
            x,
            k=self.k,
            mode=self.mode,
            include_cross=self.include_cross,
            # to make the scalar autoencoder equivariant we can refrain
            # from concatenating the input point coords to the output
            include_input=(self.mode == "vector" or self.include_coords or idx > 0),
        )

    def concat_axis(self, x):
        axis = torch.zeros(3).float()
        axis[self.break_symmetry_axis] = 1.0
        axis = axis.view(1, 1, 3, 1, 1)
        axis = axis.expand(x.shape[0], 1, 3, x.shape[-2], x.shape[-1])
        return torch.cat((x, axis), dim=1)

    def forward(self, x, get_rotation=False):
        # x is [B, N, 3]
        x = x.transpose(2, 1)  # [B, 3, N]

        batch_size = x.size(0)

        intermediate_outs = []
        for idx, conv in enumerate(self.convs[:-1]):
            x = self.get_graph_feature(x, idx)

            if idx == 0 and self.break_symmetry_axis is not None:
                x = self.concat_axis(x)

            x = conv(x)
            intermediate_outs.append(x.mean(dim=-1, keepdim=False))

        x = torch.cat(intermediate_outs, dim=1)
        x = self.final_conv(x)
        x = x.mean(dim=-1, keepdim=False)

        embedding = self.embedding(x)
        if self.mode == "vector":
            embedding = embedding.norm(dim=1)

        if get_rotation:
            return embedding, self.rotation(x)

        return embedding
