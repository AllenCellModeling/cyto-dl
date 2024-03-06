import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss as Loss


class WeightedMSELoss(Loss):
    def __init__(self, reduction="none"):
        super().__init__(None, None, reduction)
        self.reduction = reduction
        # self.weights = torch.tensor(weights).unsqueeze(0)
        self.bins = torch.linspace(-2, 2, steps=21)
        self.bins = [(i, j) for i, j, in zip(self.bins, self.bins[1:])]
        self.weights = list(torch.linspace(0, 100, steps=11)) + list(
            torch.linspace(100, 0, steps=11)
        )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        weights = torch.ones(*input.shape)
        for j, bin in enumerate(self.bins):
            bin_1 = bin[0]
            bin_2 = bin[1]
            this_mask = (input > bin_1) & (input < bin_2)
            weights[this_mask] = self.weights[j]

        loss = F.mse_loss(input, target, reduction="none")
        loss = loss * weights.type_as(loss)

        if self.reduction == "mean":
            loss = loss.mean(axis=1)
        elif self.reduction == "sum":
            loss = loss.sum(axis=1)
        return loss
