import torch
from torch import nn

from cyto_dl import utils
from .vnn import VNLinear, VNRotationMatrix

log = utils.get_pylogger(__name__)


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


class EquivTransformer(nn.Module):
    def __init__(
        self,
        encoder: dict,
        x_label: str = "pcloud",
        concat_feats: bool = False,
        num_points: int = 256,
        num_features: int = 128,
    ):
        super().__init__()
        self.encoder = encoder
        self.concat_feats = concat_feats
        self.x_label = x_label
        self.dim_feat = self.encoder.dim_feat
        self.num_points = num_points
        self.num_features = num_features

        self.pool = meanpool
        # rotation module
        self.rotation = VNRotationMatrix(self.num_points, dim=3, return_rotated=True)
        # final embedding
        self.embedding_head = VNLinear(self.num_points, self.num_features)

    def forward(self, x, get_rotation=False):
        if self.dim_feat > 0:
            coors_out, feats_out = self.encoder(
                x,
            )
        else:
            coors_out = self.encoder(x)

        x, rot = self.rotation(coors_out)
        x = self.embedding_head(x)
        x = torch.norm(x, dim=-1)

        rot = rot.mT

        if get_rotation:
            return {self.x_label: x, "rotation": rot}

        return {self.x_label: x}
