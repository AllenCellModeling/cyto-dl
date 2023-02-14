import math
from typing import Tuple

import torch
import torch.nn.functional as F


def get_rotation_matrix(v, spatial_dims=2):
    if spatial_dims == 2:
        return get_rotation_matrix_2d(v)
    return get_rotation_matrix_3d(v)


def get_rotation_matrix_2d(v):
    return torch.stack(
        (
            torch.stack((v[:, 0], v[:, 1]), dim=-1),
            torch.stack((-v[:, 1], v[:, 0]), dim=-1),
            torch.zeros(v.size(0), 2).type_as(v),
        ),
        dim=-1,
    )


def get_rotation_matrix_3d(v):
    zeros = torch.zeros(len(v)).type_as(v)
    ones = torch.ones(len(v)).type_as(v)

    return torch.stack(
        (
            torch.stack((v[:, 0], v[:, 1], zeros), dim=-1),
            torch.stack((-v[:, 1], v[:, 0], zeros), dim=-1),
            torch.stack((zeros, zeros, ones), dim=-1),
            torch.zeros(v.size(0), 3).type_as(v),
        ),
        dim=-1,
    )


def rotate_img(img, rot):
    grid = F.affine_grid(rot, img.size(), align_corners=False).type_as(img)
    return F.grid_sample(img, grid, align_corners=False)
