import numpy as np
import torch
from escnn import group, gspaces, nn
from torch.nn import functional as F


def _make_block(in_type, out_dim, gspace, irrep_ids, grid_size):
    non_linearity = nn.FourierPointwise(
        gspace,
        out_dim,
        irrep_ids,
        grid_size,
        function="p_relu",
    )

    batch_norm = nn.IIDBatchNorm1d(non_linearity.out_type, eps=1e-5)

    return nn.SequentialModule(
        nn.Linear(in_type, non_linearity.in_type),
        non_linearity,
        batch_norm,
    )


class O2SpharmEncoder(nn.EquivariantModule):
    def __init__(
        self,
        reflections=False,
        hidden_layers=[4],
        out_dim=10,
        max_spharm_band=16,
        max_hidden_band=8,
        grid_size=64,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.reflections = reflections
        self.grid_size = grid_size

        if reflections:
            O3 = group.o3_group(max_spharm_band)
            self.G = group.o2_group(max_spharm_band * 2)
            self.gspace = gspaces.no_base_space(self.G)
            _sg_id = O3._process_subgroup_id(("cone", np.pi, -1))
            _hidden_irrep_ids = [(1, k) for k in range(1, max_hidden_band + 1)]
            _repr = group.directsum(
                [O3.irrep(0, ix).restrict(_sg_id) for ix in range(max_spharm_band + 1)]
            )
        else:
            SO3 = group.so3_group(max_spharm_band)
            self.G = group.so2_group(max_spharm_band * 2)
            self.gspace = gspaces.no_base_space(self.G)
            _sg_id = SO3._process_subgroup_id((False, -1))
            _hidden_irrep_ids = [(k,) for k in range(1, max_hidden_band + 1)]
            _repr = group.directsum(
                [SO3.irrep(ix).restrict(_sg_id) for ix in range(max_spharm_band + 1)]
            )
        self.in_type = self.gspace.type(_repr)

        # this network O2 equivariant

        # our input comes in spherical harmonics, which are O3 / SO3 steerable,
        # but we want to restrict it to O2 / SO2

        block = _make_block(
            self.in_type, hidden_layers[0], self.gspace, _hidden_irrep_ids, grid_size
        )
        blocks = [block]

        for dim in hidden_layers[1:]:
            blocks.append(
                _make_block(blocks[-1].out_type, dim, self.gspace, _hidden_irrep_ids, grid_size)
            )

        if reflections:
            angle_rep = self.G.irrep(1, 1)
            flip_rep = self.G.irrep(1, 1)

            pose_type = [angle_rep, flip_rep]
        else:
            pose_type = [self.G.irrep(1)]

        embedding_type = [self.G.trivial_representation] * out_dim
        embedding_type = self.gspace.type(*embedding_type)
        pooled_type = [self.G.trivial_representation] * hidden_layers[-1]
        pooled_type = self.gspace.type(*pooled_type)
        pose_type = self.gspace.type(*pose_type)

        self.out_type = embedding_type + pose_type

        self.backbone = nn.SequentialModule(*blocks)

        self.embedding_head = nn.SequentialModule(
            nn.NormPool(blocks[-1].out_type), nn.Linear(pooled_type, embedding_type)
        )

        self.pose_head = nn.Linear(blocks[-1].out_type, pose_type)

    def forward(self, x: nn.GeometricTensor):
        y = self.backbone(x)

        y_embedding = self.embedding_head(y)
        y_pose = self.pose_head(y)

        return nn.tensor_directsum((y_embedding, y_pose))

    def evaluate_output_shape(self, input_shape: tuple):
        shape = list(input_shape)
        assert len(shape) == 2, shape
        assert shape[1] == self.in_type.size, shape
        shape[1] = self.out_type.size
        return shape
