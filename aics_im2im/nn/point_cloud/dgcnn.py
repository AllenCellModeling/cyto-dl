"""Adapted from.

- https://github.com/FlyingGiraffe/vnn/blob/master/models/utils/vn_dgcnn_util.py
  LICENSE: https://github.com/FlyingGiraffe/vnn/blob/master/LICENSE
- https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/graph_functions.py
  LICENSE: https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/LICENSE_AnTao
"""

import nnumpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .graph_functions import get_graph_features
from .vnn import VNLinear, VNLinearLeakyReLU, VNRotationMatrix


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
    ):
        super().__init__()
        self.mode = mode
        self.k = k
        self.num_features = num_features
        self.include_coords = include_coords
        self.include_cross = include_cross

        _features = [3] + hidden_features[:-1]

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

    def get_graph_features(self, x):
        return get_graph_features(
            x,
            k=self.k,
            mode=self.mode,
            include_cross=self.include_cross,
            include_coords=self.include_coords,
        )

    def forward(self, x, get_rotation=False):
        x = x.transpose(2, 1)

        batch_size = x.size(0)

        intermediate_outs = []
        for conv in self.convs[:-1]:
            x = self.get_graph_feature(x)
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
