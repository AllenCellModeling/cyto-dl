"""Adapted from.

- https://github.com/FlyingGiraffe/vnn/blob/master/models/utils/vn_dgcnn_util.py
  LICENSE: https://github.com/FlyingGiraffe/vnn/blob/master/LICENSE
- https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/graph_functions.py
  LICENSE: https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/LICENSE_AnTao
- https://github.com/SimingYan/IAE/blob/main/src/encoder/dgcnn_cls.py
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

    idx = (idx + idx_base).view(-1)
    return idx


def get_graph_features(
    x,
    k=20,
    idx=None,
    mode="scalar",
    scalar_inds=None,
    include_cross=True,
    include_input=True,
):
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
        scal = x[:, scalar_inds - 1 :, :]
        x = x[:, : scalar_inds - 1, :]
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
        feature_unit_vector = feature / torch.norm(feature, dim=1).unsqueeze(dim=1)
        scal = scal.transpose(2, 1).contiguous()
        scal = scal.view(batch_size, num_points, 1, num_scalar_points, 1).repeat(1, 1, k, 1, 1)
        scal = scal.permute(0, 3, 4, 1, 2).contiguous()
        scal = scal * feature_unit_vector
        feature = torch.cat((feature, scal), dim=1)

    return feature


def normalize_3d_coordinate(p, padding=0.1):
    """Normalize coordinate to [0, 1] for unit cube experiments. Corresponds to our 3D model.

    Args:
        p (tensor): point
        padding (float): conventional padding parameter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    p_nor = p / (1 + padding + 10e-4)  # (-0.5, 0.5)
    p_nor = p_nor + 0.5  # range (0, 1)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor


def normalize_coordinate(p, padding=0.1, plane="xz"):
    """Normalize coordinate to [0, 1] for unit cube experiments.

    Args:
        p (tensor): point
        padding (float): conventional padding parameter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        plane (str): plane feature type, ['xz', 'xy', 'yz']
    """
    if plane == "xz":
        xy = p[:, :, [0, 2]]
    elif plane == "xy":
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]

    xy_new = xy / (1 + padding + 10e-6)  # (-0.5, 0.5)
    xy_new = xy_new + 0.5  # range (0, 1)

    # f there are outliers out of the range
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new


def coordinate2index(x, reso, coord_type="2d"):
    """Normalize coordinate to [0, 1] for unit cube experiments. Corresponds to our 3D model.

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    """
    x = (x * reso).long()
    if coord_type == "2d":  # plane
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == "3d":  # grid
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index
