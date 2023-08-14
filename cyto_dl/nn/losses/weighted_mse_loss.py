import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss as Loss


class WeightedMSELoss(Loss):
    def __init__(self, reduction="none", weights=1):
        super().__init__(None, None, reduction)
        self.reduction = reduction
        self.weights = torch.tensor(weights).unsqueeze(0)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = F.mse_loss(input, target, reduction="none") * self.weights

        if self.reduction == "mean":
            loss = loss.mean(axis=1)
        elif self.reduction == "sum":
            loss = loss.sum(axis=1)
        return loss
