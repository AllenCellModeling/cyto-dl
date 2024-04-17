import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss as Loss


class FineTuneLoss2(Loss):
    def __init__(self, reduction="none"):
        super().__init__(None, None, reduction)
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = torch.where(input < 0.5, input, 0)
        input = torch.where(input > -0.5, input, 0)
        mask = torch.where(input == 0, False, True)
        target = torch.where(mask, target, 0)
        loss = F.mse_loss(input, target, reduction=self.reduction)
        return loss


# class ZeroLevelSetLoss(Loss):
#     def __init__(self, reduction="none", max_val=2):
#         super().__init__(None, None, reduction)
#         self.reduction = reduction
#         self.loss = torch.nn.L1Loss(reduction="none")

#     def forward(self, input: Tensor, target: Tensor) -> Tensor:

#         input = torch.where(input < 0, -1, input)
#         input = torch.where(input > 0, 1, input)

#         target = torch.where(target < 0, -1, target)
#         target = torch.where(target > 0, 1, target)


#         if self.reduction == "mean":
#             loss = loss.mean(axis=1)
#         elif self.reduction == "sum":
#             loss = loss.sum(axis=1)
#         return loss
