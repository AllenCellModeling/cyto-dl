import torch.nn.functional as F
from torch.nn.modules.loss import _Loss as Loss


class CosineLoss(Loss):
    def __init__(self, reduction="mean"):
        super().__init__(None, None, reduction)

    def forward(self, input, target):
        # sum per input-element log loss
        loss = 1 - F.cosine_similarity(input, target)

        # reduce across batch dimension
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        else:
            raise NotImplementedError(f"Unavailable reduction type: {self.reduction}")
