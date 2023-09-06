"""Adapted from:

- https://github.com/FlyingGiraffe/vnn-neural-implicits/blob/master/im2mesh/layers_equi.py
- https://github.com/FlyingGiraffe/vnn-neural-implicits/blob/master/im2mesh/encoder/vnn2.py

License: https://github.com/FlyingGiraffe/vnn-neural-implicits/blob/master/LICENSE
"""

import torch
from torch import nn
from torch.nn import functional as F

EPS = 1e-12


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        """
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        """
        out = self.linear(x.transpose(1, -1)).transpose(1, -1)

        return out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super().__init__()
        self.dim = dim
        if dim in (3, 4):
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        """
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        """
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn

        return x


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super().__init__()
        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        """
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        """
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d)
        )
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dim=5,
        share_nonlinearity=False,
        use_batchnorm=True,
        negative_slope=0.2,
        eps=1e-12,
    ):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # Conv
        self.map_to_feat = VNLinear(in_channels, out_channels)

        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.batchnorm = VNBatchNorm(out_channels, dim=dim)

        if share_nonlinearity:
            self.map_to_dir = VNLinear(in_channels, 1)
        else:
            self.map_to_dir = VNLinear(in_channels, out_channels)

    def forward(self, x):
        """
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        """
        # Conv
        p = self.map_to_feat(x)

        # InstanceNorm
        if self.use_batchnorm:
            p = self.batchnorm(p)

        # LeakyReLU
        d = self.map_to_dir(x)
        dotprod = (p * d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1 - self.negative_slope) * (
            mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + self.eps)) * d)
        )
        return x_out


class VNRotationMatrix(nn.Module):
    def __init__(
        self,
        in_channels,
        share_nonlinearity=False,
        use_batchnorm=False,
        eps=1e-12,
        dim=4,
        return_rotated=True,
    ):
        super().__init__()
        self.eps = eps
        self.use_batchnorm = use_batchnorm
        self.dim = dim

        self.vn1 = VNLinearLeakyReLU(
            in_channels,
            in_channels // 2,
            dim=self.dim,
            share_nonlinearity=share_nonlinearity,
            use_batchnorm=use_batchnorm,
            eps=eps,
        )
        self.vn2 = VNLinearLeakyReLU(
            in_channels // 2,
            in_channels // 4,
            dim=self.dim,
            share_nonlinearity=share_nonlinearity,
            use_batchnorm=use_batchnorm,
            eps=eps,
        )
        self.vn_lin = VNLinear(in_channels // 4, 2)
        self.return_rotated = return_rotated

    def forward(self, x):
        """
        x: point features of shape [B, N_feat, 3]
        """
        z = self.vn1(x)
        z = self.vn2(z)
        z = self.vn_lin(z)

        # produce two orthogonal vectors
        v1 = z[:, 0, :]

        v1_norm = torch.sqrt((v1 * v1).sum(1, keepdims=True))
        u1 = v1 / (v1_norm + self.eps)

        v2 = z[:, 1, :]
        v2 = v2 - (v2 * u1).sum(1, keepdim=True) * u1

        v2_norm = torch.sqrt((v2 * v2).sum(1, keepdims=True))
        u2 = v2 / (v2_norm + self.eps)

        # compute the cross product of the two output vectors
        u3 = torch.cross(u1, u2)
        rot = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)

        if self.return_rotated:
            if self.dim == 4:
                x_std = torch.einsum("bijm,bjkm->bikm", x, rot)
            elif self.dim == 3:
                x_std = torch.einsum("bij,bjk->bik", x, rot)
            elif self.dim == 5:
                x_std = torch.einsum("bijmn,bjkmn->bikmn", x, rot)
            return x_std, rot
        return rot
