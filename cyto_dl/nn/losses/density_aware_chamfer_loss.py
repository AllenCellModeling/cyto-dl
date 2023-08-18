"""
Adapted from: https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/chamfer_distance.py
LICENSE: https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/LICENSE_AnTao
"""

import torch
import torch.nn as nn

from cyto_dl.nn.losses.metrics import cd, emd

# install these


class DensityAwareChamferLoss(nn.Module):
    def __init__(
        self,
        alpha,
        n_lambda,
        **kwargs,
    ):
        super().__init__()
        self.alpha = alpha
        self.n_lambda = n_lambda

    def calc_dcd_full(
        self,
        x,
        gt,
        return_raw=False,
        non_reg=False,
        only_loss=True,
    ):
        x = x.float()
        gt = gt.float()
        alpha = self.alpha
        n_lambda = self.n_lambda
        batch_size, n_x, _ = x.shape
        batch_size, n_gt, _ = gt.shape
        assert x.shape[0] == gt.shape[0]

        if non_reg:
            frac_12 = max(1, n_x / n_gt)
            frac_21 = max(1, n_gt / n_x)
        else:
            frac_12 = n_x / n_gt
            frac_21 = n_gt / n_x

        cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True)
        # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
        # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
        # dist2 and idx2: vice versa
        exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

        count1 = torch.zeros_like(idx2)
        count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
        weight1 = count1.gather(1, idx1.long()).float().detach() ** n_lambda
        weight1 = (weight1 + 1e-6) ** (-1) * frac_21
        loss1 = (1 - exp_dist1 * weight1).mean(dim=1)

        count2 = torch.zeros_like(idx1)
        count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
        weight2 = count2.gather(1, idx2.long()).float().detach() ** n_lambda
        weight2 = (weight2 + 1e-6) ** (-1) * frac_12
        loss2 = (1 - exp_dist2 * weight2).mean(dim=1)

        loss = (loss1 + loss2) / 2
        if only_loss:
            return loss

        res = [loss, cd_p, cd_t]
        if return_raw:
            res.extend([dist1, dist2, idx1, idx2])

        return res

    def forward(self, gts, preds):
        return self.calc_dcd_full(preds, gts)


def calc_cd(
    output, gt, calc_f1=False, return_raw=False, normalize=False, separate=False
):
    # cham_loss = dist_chamfer_3D.chamfer_3DDist()
    cham_loss = cd()
    dist1, dist2, idx1, idx2 = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = dist1.mean(1) + dist2.mean(1)

    if separate:
        res = [
            torch.cat(
                [
                    torch.sqrt(dist1).mean(1).unsqueeze(0),
                    torch.sqrt(dist2).mean(1).unsqueeze(0),
                ]
            ),
            torch.cat([dist1.mean(1).unsqueeze(0), dist2.mean(1).unsqueeze(0)]),
        ]
    else:
        res = [cd_p, cd_t]
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2)
        res.append(f1)
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return res
