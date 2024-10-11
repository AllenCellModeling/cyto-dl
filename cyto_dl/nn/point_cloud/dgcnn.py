"""Adapted from.

- https://github.com/FlyingGiraffe/vnn/blob/master/models/utils/vn_dgcnn_util.py
  LICENSE: https://github.com/FlyingGiraffe/vnn/blob/master/LICENSE
- https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/graph_functions.py
  LICENSE: https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/LICENSE_AnTao
"""

import torch
from torch import nn
from torch_scatter import scatter_max, scatter_mean

from cyto_dl import utils

from .graph_functions import (
    coordinate2index,
    get_graph_features,
    normalize_3d_coordinate,
    normalize_coordinate,
)
from .vnn import VNLeakyReLU, VNLinear, VNLinearLeakyReLU, VNRotationMatrix

log = utils.get_pylogger(__name__)


def _make_conv(
    in_features,
    out_features,
    mode="scalar",
    scale_in=1,
    add_in=0,
    include_symmetry=0,
    scale_out=1,
    final=False,
):
    in_features = in_features * scale_in + include_symmetry + add_in
    out_features = out_features * scale_out

    if mode == "vector":
        return VNLinearLeakyReLU(
            in_features,
            out_features,
            use_batchnorm=False,
            negative_slope=0,
            dim=(4 if final else 5),
        )
    elif mode == "vector2":
        return nn.Sequential(
            VNLeakyReLU(in_features, negative_slope=0.0, share_nonlinearity=False),
            VNLinear(in_features, out_features),
        )

    conv = nn.Conv1d if final else nn.Conv2d
    batch_norm = nn.BatchNorm1d if final else nn.BatchNorm2d

    conv = conv(in_features, out_features, kernel_size=1, bias=False)
    # conv.weight.data.fill_(0.01)

    return nn.Sequential(
        conv,
        batch_norm(out_features),
        nn.LeakyReLU(negative_slope=0.2),
    )


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


class DGCNN(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_dim=64,
        k=20,
        mode="scalar",
        hidden_conv2d_channels=[
            64,
            64,
            64,
            64,
        ],  # [64, 64, 64, 64] for vector and scalar
        hidden_conv1d_channels=[
            512,
            512,
        ],  # [512, 1024, 512, num_features] for dgcnn cls (where 1024 is latent dim) and [512, 512] for normal dgcnn, [64, num_feats] for vector
        scalar_inds=None,
        include_cross=True,
        include_coords=True,
        symmetry_breaking_axis=None,
        padding=0.1,
        reso_plane=64,
        reso_grid=None,
        plane_type=["xz", "xy", "yz"],
        generate_grid_feats=False,
        scatter_type="max",
        x_label="pcloud",
    ):
        super().__init__()
        self.k = k
        self.x_label = x_label
        self.num_features = num_features
        self.scatter_type = scatter_type
        self.include_coords = include_coords
        self.hidden_conv2d_channels = hidden_conv2d_channels
        self.hidden_conv1d_channels = hidden_conv1d_channels
        self.include_cross = include_cross
        self.hidden_dim = hidden_dim
        self.scalar_inds = scalar_inds
        self.mode = mode
        self.padding = padding
        self.reso_plane = reso_plane
        self.reso_grid = reso_grid
        self.plane_type = plane_type
        self.generate_grid_feats = generate_grid_feats
        self.unet3d = None
        self.unet = None
        self.symmetry_breaking_axis = symmetry_breaking_axis

        include_symmetry = 0
        if self.symmetry_breaking_axis is not None:
            include_symmetry = 1

        self.mode = mode
        self.init_features = 1 if self.mode == "vector" else 3
        self.final_conv = []
        if self.scalar_inds:
            _scalar_scale = 2
        else:
            _scalar_scale = 1

        # first conv
        convs = [
            _make_conv(
                self.init_features,
                self.hidden_dim,
                self.mode,
                scale_in=(1 + include_coords + include_cross) * _scalar_scale,
                include_symmetry=include_symmetry,
            )
        ]

        if self.mode == "vector":
            convs += [
                nn.Sequential(
                    VNLinear(self.hidden_dim, self.hidden_dim * 2),
                    VNLeakyReLU(
                        2 * self.hidden_dim,
                        negative_slope=0.0,
                        share_nonlinearity=False,
                    ),
                    VNLinear(2 * self.hidden_dim, self.hidden_dim),
                )
            ]
            _scale_in_list = [2 for i in range(len(hidden_conv2d_channels) - 1)]
            _scale_out_list = [1 for i in range(len(hidden_conv2d_channels) - 1)]
            _mode = "vector2"
            _prev_slice = -1
            self.pool = meanpool
            # rotation module
            self.rotation = VNRotationMatrix(self.num_features, dim=3, return_rotated=True)
            # final embedding
            self.embedding_head = VNLinear(self.num_features, self.num_features)
        else:
            _scale_in_list = [2] + [(i + 1) * 2 for i in range(len(hidden_conv2d_channels) - 2)]
            _scale_out_list = [1] + [(i + 1) * 2 for i in range(len(hidden_conv2d_channels) - 2)]
            _prev_slice = -3
            _mode = "scalar"
            self.pool = maxpool
            self.embedding_head = nn.Linear(self.hidden_conv1d_channels[-1], self.num_features)

        for j, (c_1, c_2) in enumerate(
            zip(self.hidden_conv2d_channels[:-1], self.hidden_conv2d_channels[1:])
        ):
            convs += [
                _make_conv(
                    c_1,
                    c_2,
                    mode=_mode,
                    scale_in=_scale_in_list[j],
                    scale_out=_scale_out_list[j],
                )
            ]
        _prev_in = 0
        for j, (c_1, c_2) in enumerate(
            zip(self.hidden_conv1d_channels[:-1], self.hidden_conv1d_channels[1:])
        ):
            if j == 1:
                self.final_conv += _make_conv(c_1, c_2, final=True, add_in=_prev_in, mode=_mode)
            else:
                self.final_conv += _make_conv(c_1, c_2, final=True, mode=_mode)
            _prev_in = self.final_conv[_prev_slice].in_channels

        self.convs = nn.ModuleList(convs)
        self.final_conv = nn.ModuleList(self.final_conv)

        if self.scatter_type == "max":
            self.scatter = scatter_max
        else:
            self.scatter = scatter_mean

    def get_graph_features(self, x, idx):
        return get_graph_features(
            x,
            k=self.k,
            mode=self.mode,
            scalar_inds=self.scalar_inds,
            include_cross=self.include_cross,
            # to make the scalar autoencoder equivariant we can refrain
            # from concatenating the input point coords to the output
            include_input=(self.mode == "vector" or self.include_coords or idx > 0),
        )

    def concat_axis(self, x, axis):
        if isinstance(axis, int):
            # symmetry axis is aligned with one of the X,Y,Z axes
            _axis = torch.zeros(3).float()
            _axis[axis] = 1.0
            _axis = _axis.view(1, 1, 3, 1, 1)
            # _axis = _axis.view(1, 3, 1, 1, 1)
        else:
            # per-element symmetry axis (e.g. moving direction)
            _axis = axis.view(x.shape[0], 1, 3, 1, 1)
            # _axis = axis.view(x.shape[0], 3, 1, 1, 1)

        _axis = _axis.expand(x.shape[0], 1, 3, x.shape[-2], x.shape[-1])
        # _axis = _axis.expand(x.shape[0], 3, 1, x.shape[-2], x.shape[-1])
        _axis = _axis.type_as(x)

        return torch.cat((x, _axis), dim=1)

    def _generate_plane_features(self, points, cond, plane="xz"):
        view_dims1 = (points.size(0), self.num_features, self.reso_plane**2)
        view_dims2 = (
            points.size(0),
            self.num_features,
            self.reso_plane,
            self.reso_plane,
        )

        permute_dims1 = (0, 2, 1)
        # acquire indices of features in plane
        xy = normalize_coordinate(
            points.clone(), plane=plane, padding=self.padding
        )  # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = cond.new_zeros(*view_dims1)
        cond = cond.permute(*permute_dims1)  # B x 512 x T
        fea_plane = scatter_mean(cond, index, out=fea_plane).reshape(
            *view_dims2
        )  # sparse matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def _generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type="3d")
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.num_features, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid).reshape(
            p.size(0), self.num_features, self.reso_grid, self.reso_grid, self.reso_grid
        )  # sparse matrix (B x 512 x reso x reso)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid

    def _pool_local(self, xy, index, c):
        _, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == "grid":
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid**3)
            else:
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane**2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, x, get_rotation=False):
        input_pc = x.clone()
        # x is [B, N, 3]
        x = x.transpose(2, 1)  # [B, 3, N]
        num_points = x.shape[-1]
        intermediate_outs = []
        for idx, conv in enumerate(self.convs):
            if (idx == 0 and self.mode == "vector") or (self.mode == "scalar"):
                x = self.get_graph_features(x, idx)

            if idx == 0 and self.symmetry_breaking_axis is not None:
                if isinstance(self.symmetry_breaking_axis, int):
                    x = self.concat_axis(x, self.symmetry_breaking_axis)
                assert x.size(1) == 4

            pre_x = conv(x)

            if (len(pre_x.size()) < 5) and (self.mode == "vector") and (idx > 0):
                if (idx > 0) & (idx < len(self.convs) - 1):
                    x = self.pool(pre_x, dim=-1, keepdim=True).expand(pre_x.size())
                    x = torch.cat([x, pre_x], dim=1)
                else:
                    x = pre_x
            else:
                x = self.pool(pre_x, dim=-1)

            intermediate_outs.append(x)

        x = torch.cat(intermediate_outs, dim=1)
        if self.mode == "scalar":
            _pool_ind = 2
        else:
            _pool_ind = 1

        for j, conv in enumerate(self.final_conv):
            x = conv(x)
            if (j == _pool_ind) and (self.generate_grid_feats):
                x = self.pool(x, dim=-1, keepdim=True)
                pre_repeat = x.clone()  # this is batch_size, feat dims
                if len(x.shape) == 3:
                    x = x.repeat(1, 1, num_points)
                else:
                    x = x.repeat(1, 1, 1, num_points)
                x = torch.cat((x, *intermediate_outs), dim=1)
            elif (j == _pool_ind) and (not self.generate_grid_feats):
                x = self.pool(x, dim=-1)

        rot = torch.zeros(1).type_as(x)
        if self.generate_grid_feats:
            if len(x.shape) == 4:
                _, rot = self.rotation(pre_repeat.squeeze(dim=-1))
                rot = rot.mT
                x = torch.norm(x, dim=-2)
            x = x.permute(0, 2, 1).contiguous()

            fea = {}
            if "grid" in self.plane_type:
                fea["grid"] = self._generate_grid_features(input_pc, x)
            if "xz" in self.plane_type:
                fea["xz"] = self._generate_plane_features(input_pc, x, plane="xz")
            if "xy" in self.plane_type:
                fea["xy"] = self._generate_plane_features(input_pc, x, plane="xy")
            if "yz" in self.plane_type:
                fea["yz"] = self._generate_plane_features(input_pc, x, plane="yz")
            return {self.x_label: pre_repeat, "rotation": rot, "grid_feats": fea}

        if self.mode == "vector":
            x, rot = self.rotation(x)
            x = self.embedding_head(x)
            x = torch.norm(x, dim=-1)
            rot = rot.mT

        if self.mode == "scalar":
            x = self.embedding_head(x)

        if get_rotation:
            return {self.x_label: x, "rotation": rot}

        return {self.x_label: x}
