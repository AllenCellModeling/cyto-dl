"""
Adapted from: https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/chamfer_distance.py
LICENSE: https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/LICENSE_AnTao
"""

import torch
import torch.nn as nn


class ChamferLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = x.pow(2).sum(dim=-1)
        yy = y.pow(2).sum(dim=-1)
        zz = torch.bmm(x, y.transpose(2, 1))
        rx = xx.unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy.unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P

    def batch_pairwise_dist2(self, x, y):
        P = torch.cdist(x, y, p=2)
        return P

    def forward(self, gts, preds):
        P = self.batch_pairwise_dist2(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins, axis=1)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins, axis=1)
        return loss_1 + loss_2
