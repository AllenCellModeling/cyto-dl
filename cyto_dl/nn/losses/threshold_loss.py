import torch
from numpy.typing import ArrayLike
from torch import nn


class ThresholdLoss(nn.Module):
    def __init__(self, loss_fn, threshold: float = 0.0, above: bool = True):
        """Wrapper Loss that thresholds the target before computing the loss given by loss_fn.

        Parameters
        ----------
            loss_fn
                Loss function
            threshold: float = 0.0
                Threshold value
            above: bool = True
                Whether to threshold above or below
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.threshold = threshold
        self.above = above

    def __call__(self, input, target):
        if self.above:
            target = (target >= self.threshold).type_as(target)
        else:
            target = (target <= self.threshold).type_as(target)

        return self.loss_fn(input, target)
