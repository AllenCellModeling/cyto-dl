import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss as Loss


class GaussianNLLLoss(Loss):
    def __init__(self, mean_dims=None, eps=1e-10):
        super().__init__(None, None, "none")
        self.mean_dims = tuple(mean_dims)
        self.eps = 1e-10

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.mean_dims is None:
            mean_dims = [_ for _ in range(len(input.shape))]
        else:
            mean_dims = self.mean_dims

        sigma = ((target - input) ** 2).mean(mean_dims, keepdim=True).sqrt()
        log_sigma = (sigma + self.eps).log().detach()

        loss = (
            (
                0.5 * torch.pow((target - input) / log_sigma.exp(), 2)
                + log_sigma
                + 0.5 * np.log(2 * np.pi)
            )
            .reshape(input.shape[0], -1)
            .sum(dim=1, keepdim=True)
        )

        return loss
