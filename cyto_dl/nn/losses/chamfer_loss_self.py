"""
Adapted from: https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/chamfer_distance.py
LICENSE: https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/LICENSE_AnTao
"""

import torch
import torch.nn as nn


class ChamferLossSelf(nn.Module):
    def __init__(
        self,
        alpha: float = 1,
        **kwargs,
    ):
        super().__init__()
        self.alpha = alpha

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
        if isinstance(gts, list):
            gts = torch.stack(gts, dim=0)
        # P = self.batch_pairwise_dist2(gts, preds)
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins, axis=1)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins, axis=1)

        P1 = self.batch_pairwise_dist(gts, gts)
        P2 = self.batch_pairwise_dist(preds, preds)

        for i in range(P1.shape[0]):
            P1[i].fill_diagonal_(500000)
            P2[i].fill_diagonal_(500000)
        mins1, _ = torch.min(P1, 1)
        mins2, _ = torch.min(P2, 1)
        mins1_sorted, _ = torch.sort(mins1)
        mins2_sorted, _ = torch.sort(mins2)

        min_ps = min(mins1_sorted.shape[1], mins2_sorted.shape[1])

        loss_ordering = torch.sum(
            (mins1_sorted[:, :min_ps] - mins2_sorted[:, :min_ps]) ** 2, axis=1
        )

        return loss_1 + loss_2 + self.alpha * loss_ordering
