import math
from typing import Tuple

import torch
import torch.nn.functional as F


class RotationModule:
    def __init__(self, group, spatial_dims, pad_value=0.0, eps=1e-6):
        if group not in ("so2", "so3"):
            raise ValueError(f"Group must be one of ('so2', 'so3'), got {group}")

        self.group = group
        self.spatial_dims = spatial_dims
        self.eps = eps
        self.pad_value = pad_value

    def compute_rotation_matrix(self, pose):
        if self.group == "so2":
            return get_rotation_matrix_so2(pose, self.spatial_dims)

        return get_rotation_matrix_so3(pose, eps=self.eps)

    def __call__(self, img, pose, R=None):
        if R is None:
            R = self.compute_rotation_matrix(pose)

        if self.spatial_dims == 3:
            # assume data comes in -Z -Y X format, to match `escnn`.
            # the rotation matrix comes in XYZ form, so I transform it
            # to X -Y -Z
            R[:, -2:, :] = R[:, -2:, :] * -1
            R[:, :, -2:] = R[:, :, -2:] * -1

        if self.spatial_dims == 2:
            # affine_grid and grid_sample rotate the grid, not the signal.
            # to rotate the signal by alpha, we want to rotate the grid
            # by -alpha, so we make that correction here.
            R[:, -1, :] = R[:, -1, :] * -1
            R[:, :, -1] = R[:, :, -1] * -1

        # add a displacement vector of zeros to the rotation matrix
        disp = torch.tensor(0).expand(len(img), self.spatial_dims, 1).type_as(img)
        A = torch.cat((R, disp), dim=2)

        grid = F.affine_grid(A, img.size(), align_corners=False).type_as(img)

        y = F.grid_sample(img - self.pad_value, grid, align_corners=False)
        return y + self.pad_value


def get_rotation_matrix_so2(pose, spatial_dims):
    """Computes a (batch of) rotation matrix of the SO(2) group, in either 2d or 3d. In 3d,
    rotation is assumed to be about the Z axis (first axis).

    Parameters
    ----------
    pose: torch.Tensor
        A (batch of) equivariant 2d vector(s), of the form (cos(theta), sin(theta))

    spatial_dims: int
        Indicates whether it's 2d or 3d

    Returns
    -------
    The (batch of) rotation matrix
    """
    if len(pose.shape) != 2:
        v = pose.unsqueeze(0)
    else:
        v = pose

    if spatial_dims == 2:
        R = torch.stack(
            (
                torch.stack((v[:, 0], -v[:, 1]), dim=1),
                torch.stack((v[:, 1], v[:, 0]), dim=1),
            ),
            dim=1,
        )
    else:
        zeros = torch.tensor(0).type_as(v).expand(len(v))
        ones = torch.tensor(1).type_as(v).expand(len(v))

        R = torch.stack(
            (
                torch.stack((v[:, 0], -v[:, 1], zeros), dim=1),
                torch.stack((v[:, 1], v[:, 0], zeros), dim=1),
                torch.stack((zeros, zeros, ones), dim=1),
            ),
            dim=1,
        )

    if len(pose.shape) != 2:
        return R.squeeze(0)
    return R


def get_rotation_matrix_so3(z, eps=1e-6):
    """Computes a (batch of) rotation matrix of the SO(3) group.

    Parameters
    ----------
    z: torch.Tensor
        A batch of pairs of equivariant vectors, from which a rotation matrix
        is inferred

    eps: float
        Precision

    Returns
    -------
    The (batch of) rotation matrix
    """

    # produce  unit vector
    v1 = z[:, 0, :]
    u1 = v1 / (v1.norm(dim=1, keepdim=True) + eps)

    # produce a second unit vector, orthogonal to the first one
    v2 = z[:, 1, :]
    v2 = v2 - (v2 * u1).sum(1, keepdim=True) * u1
    u2 = v2 / (v2.norm(dim=1, keepdim=True) + eps)

    # produce a third orthogonal vector, as the cross product of the first two
    u3 = torch.cross(u1, u2)
    rot = torch.stack([u1, u2, u3], dim=1)

    return rot
