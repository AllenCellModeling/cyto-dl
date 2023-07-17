import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss as Loss


class GaussianNLLLoss(Loss):
    def __init__(self, reduction="none", var=None, eps=1e-12, full=False):
        super().__init__(None, None, reduction)
        self.reduction = reduction
        self.var = torch.tensor(list(var)) if var else None
        self.eps = eps
        self.full = full or (var is not None)

        if self.full:
            self.constant = 0.0
        else:
            self.constant = 0.5 * torch.log(torch.max(self.var, torch.tensor(eps)))

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.constant.device != input.device:
            self.constant = self.constant.to(input.device)

        if self.var:
            if self.var.device != input.device:
                self.var = self.var.to(input.device)
            var = self.var.repeat(len(input), 1)
            mean = input
        else:
            mean, var = torch.split(input, input.shape[1] // 2, dim=1)
            var = F.softplus(var).add(1e-3)

        return (
            F.gaussian_nll_loss(
                mean,
                target,
                var,
                full=self.full,
                eps=self.eps,
                reduction=self.reduction,
            )
            - self.constant
        )
