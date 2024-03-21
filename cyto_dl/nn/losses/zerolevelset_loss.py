import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss as Loss


class ZeroLevelSetLoss(Loss):
    def __init__(self, reduction="none", max_val=2):
        super().__init__(None, None, reduction)
        self.reduction = reduction
        self.max_val = max_val

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = torch.abs(self.max_val - torch.abs(input))
        target = torch.abs(self.max_val - torch.abs(target))
        loss = F.mse_loss(input, target, reduction="none")

        if self.reduction == "mean":
            loss = loss.mean(axis=1)
        elif self.reduction == "sum":
            loss = loss.sum(axis=1)
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
