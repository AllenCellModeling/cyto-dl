import torch
from torch.distributions import Weibull
from torch.nn.modules.loss import _Loss as Loss


def weibull_log_probs(beta, alpha, target, eps):
    alpha = alpha + eps
    return (
        torch.log(beta)
        - torch.log(alpha)
        + (beta - 1) * (torch.log(target) - torch.log(alpha))
        - beta * (target / alpha)
    )


class WeibullLogLoss(Loss):
    def __init__(self, reduction="sum", mode="explicit", eps=1e-10):
        super().__init__(None, None, reduction)
        self.mode = mode
        self.eps = eps

    def forward(self, a, b, target):
        if self.mode == "explicit":
            log_probs = weibull_log_probs(b, a, target, self.eps)
        else:
            log_probs = Weibull(a, b).log_prob(target)

        # reduce across batch dimension
        if self.reduction == "none":
            return -log_probs
        elif self.reduction == "sum":
            return -log_probs.sum()
        elif self.reduction == "mean":
            return -log_probs.mean()
        else:
            raise NotImplementedError(f"Unavailable reduction type: {self.reduction}")
