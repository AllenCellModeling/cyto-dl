"""Adapted from.

- https://github.com/FlyingGiraffe/vnn/blob/master/models/utils/vn_dgcnn_util.py
  LICENSE: https://github.com/FlyingGiraffe/vnn/blob/master/LICENSE
- https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/graph_functions.py
  LICENSE: https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/LICENSE_AnTao
"""

import torch
from monai.networks.layers.simplelayers import Flatten
from torch import nn
from torch.nn import functional as F

from .graph_functions import get_graph_feature


def _make_conv(in_features, out_features, mode="scalar"):
    if mode == "vector":
        pass

    return nn.Sequential(
        nn.Conv2d(in_features * 2, out_features, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_features),
        nn.LeakyReLU(negative_slope=0.2),
    )


class DGCNN(nn.Module):
    def __init__(self, num_features, hidden_features=[64, 128, 256, 512], k=20, mode="scalar"):
        super().__init__()
        self.mode = mode
        self.k = k
        self.num_features = num_features

        _features = [3] + hidden_features

        convs = [
            _make_conv(in_features, out_features, mode)
            for in_features, out_features in zip(_features[:-1], _features[1:])
        ]

        self.clustering = None
        self.lin_features_len = 512
        if (self.num_features < self.lin_features_len) or (
            self.num_features > self.lin_features_len
        ):
            self.flatten = Flatten()
            self.embedding = nn.Linear(self.lin_features_len, self.num_features, bias=False)

    def forward(self, x):
        # print(x.shape)
        # print(x)
        x = x.transpose(2, 1)

        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x0 = self.conv5(x)
        x = x0.max(dim=-1, keepdim=False)[0]
        feat = x.unsqueeze(1)

        if (self.num_features < self.lin_features_len) or (
            self.num_features > self.lin_features_len
        ):
            x = self.flatten(feat)
            features = self.embedding(x)
        else:
            features = torch.reshape(torch.squeeze(feat), (batch_size, 512))

        return features
