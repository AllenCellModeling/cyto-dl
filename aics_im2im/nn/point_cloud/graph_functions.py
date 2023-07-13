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

    idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    if idx_base.device != idx.device:
        idx_base = idx_base.to(idx.device)

    idx = idx + idx_base
    idx = idx.view(-1)

    return idx


def get_graph_features(x, k=20, idx=None, mode="scalar", scalar_inds=None,include_cross=True, include_input=True):
    batch_size = x.shape[0]
    num_points = x.shape[-1]
    assert len(x.shape) in (3, 4)
    if len(x.shape) == 4:
        assert mode == "vector"

    if mode == "vector":
        if len(x.size()) == 3:
            x = x.unsqueeze(1)  # [B, 1, 3, num_points]

    x = x.view(batch_size, -1, num_points)

    if scalar_inds:
        scals = x[:,scalar_inds-1:,: ]
        x = x[:,:scalar_inds-1,:]
        num_scalar_points = scal.size(1)

    if idx is None:
        idx = knn(x, k=k)

    num_dims = x.size(1)

    if mode == "vector":
        num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]

    if mode == "vector":
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

    if mode == "vector" and include_cross:
        cross = torch.cross(feature, x, dim=-1)
        feature = torch.cat((feature - x, cross), dim=3)
    else:
        feature = feature - x

    if include_input:
        feature = torch.cat((feature, x), dim=3)

    feature = feature.permute(*permute_dims).contiguous()

    if scalar_inds:
        feature_unit_vector = feature/torch.norm(feature, dim=1).unsqueeze(dim=1)
        scal = scal.transpose(2,1).contiguous()
        scal = scal.view(batch_size, num_points, 1, num_scalar_points, 1).repeat(1, 1, k, 1, 1)
        scal = scal.permute(0,3,4,1,2).contiguous()
        scal = scal*feature_unit_vector
        feature = torch.cat((feature, scal), dim=1)
        
    return feature
