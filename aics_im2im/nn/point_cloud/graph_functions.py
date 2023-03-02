"""Adapted from.

- https://github.com/FlyingGiraffe/vnn/blob/master/models/utils/vn_dgcnn_util.py
  LICENSE: https://github.com/FlyingGiraffe/vnn/blob/master/LICENSE
- https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/graph_functions.py
  LICENSE: https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/LICENSE_AnTao
"""

import torch


def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    else:
        idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx


def get_graph_features(x, k=20, idx=None, vectors=False, cross=True):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)

    _, num_dims, _ = x.size()

    if vectors:
        num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]

    if vectors:
        feature_view_dims = (batch_size, num_points, k, num_dims, 3)
        x_view_dims = (batch_size, num_points, 1, num_dims, 3)
        repeat_dims = (1, 1, k, 1, 1)
        permute_dims = (0, 3, 4, 1, 2)
    else:
        feature_view_dims = (batch_size, num_points, k, num_dims)
        x_view_dims = (batch_size, num_points, 1, num_dims)
        repeat_dims = (1, 1, k, 1)
        permute_dims = (0, 3, 1, 2)

    feature = feature.view(*feature_view_dims)
    x = x.view(*x_view_dims).repeat(*repeat_dims)

    if vectors and cross:
        cross = torch.cross(feature, x, dim=-1)
        feature = torch.cat((feature - x, x, cross), dim=3)
    else:
        feature = torch.cat((feature - x, x), dim=3)

    return feature.permute(*permute_dims).contiguous()
