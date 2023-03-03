"""Adapted from:

- https://github.com/FlyingGiraffe/vnn-neural-implicits/blob/master/im2mesh/layers_equi.py
- https://github.com/FlyingGiraffe/vnn-neural-implicits/blob/master/im2mesh/encoder/vnn2.py

License: https://github.com/FlyingGiraffe/vnn-neural-implicits/blob/master/LICENSE
"""

import torch
from torch import nn
from torch.nn import functional as F


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        linear = nn.Linear(in_channels, out_channels, bias=False)
        self.weight = nn.Parameter(linear.weight.data)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        """
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        """

        n_squeeze = 0
        if len(x.shape) == 3:
            x = x.unsqueeze(-1).unsqueeze(-1)
        elif len(x.shape) == 4:
            x = x.unsqueeze(-1)
            n_squeeze = 1

        orig_shape = x.shape
        stacked = x.view(x.shape[0], x.shape[1] * 3, x.shape[-2], x.shape[-1])
        out = F.conv2d(
            stacked,
            torch.kron(self.weight, torch.eye(3)).unsqueeze(-1).unsqueeze(-1),
            bias=None,
        )

        out = out.view(orig_shape[0], self.out_channels, 3, orig_shape[-2], orig_shape[-1])

        for _ in range(n_squeeze):
            out = out.squeeze(-1)

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
        norm = torch.norm(x, dim=2)
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn

        return x


class VNLeakyReLU(nn.Module):
    def __init__(
        self, in_channels, out_channels, share_nonlinearity=False, negative_slope=0.2, eps=1e-6
    ):
        super().__init__()
        if share_nonlinearity:
            self.map_to_dir = VNLinear(in_channels, 1)
        else:
            self.map_to_dir = VNLinear(in_channels, out_channels)

        self.negative_slope = negative_slope
        self.in_channels = in_channels
        self.eps = eps

    def forward(self, x, x_dir=None):
        """
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        """

        x_dir = x if x_dir is None else x_dir
        d = self.map_to_dir(x_dir)

        dotprod = (x * d).sum(2, keepdim=True)
        d_norm_sq = (d * d).sum(2, keepdim=True)

        return torch.where(
            dotprod >= 0, x, x - (1 - self.negative_slope) * (dotprod / (d_norm_sq + self.eps)) * d
        )


class VNLinearLeakyReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dim=5,
        share_nonlinearity=False,
        use_batchnorm=True,
        negative_slope=0.2,
        eps=1e-6,
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

        self.leaky_relu = VNLeakyReLU(
            in_channels, out_channels, share_nonlinearity, negative_slope, eps
        )

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
        return self.leaky_relu(p, x)


class VNRotationMatrix(nn.Module):
    def __init__(
        self,
        in_channels,
        dim=5,
        share_nonlinearity=False,
        use_batchnorm=True,
        eps=1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.use_batchnorm = use_batchnorm

        self.vn1 = VNLinearLeakyReLU(
            in_channels,
            in_channels // 2,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            use_batchnorm=use_batchnorm,
            eps=eps,
        )
        self.vn2 = VNLinearLeakyReLU(
            in_channels // 2,
            in_channels // 4,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            use_batchnorm=use_batchnorm,
            eps=eps,
        )
        self.vn_lin = VNLinear(in_channels // 4, 2)

    def forward(self, x):
        """
        x: point features of shape [B, N_feat, 3]
        """
        z = self.vn1(x)
        z = self.vn2(z)
        z = self.vn_lin(z)

        # produce two orthogonal vectors
        v1 = z[:, 0, :]

        v1_norm = torch.norm(v1, dim=1, keepdim=True)
        u1 = v1 / (v1_norm + self.eps)

        v2 = z[:, 1, :]
        v2 = v2 - (v2 * u1).sum(1, keepdim=True) * u1

        v2_norm = torch.norm(v2, dim=1, keepdim=True)
        u2 = v2 / (v2_norm + self.eps)

        # compute the cross product of the two output vectors
        u3 = torch.cross(u1, u2)
        rot = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)

        return rot
