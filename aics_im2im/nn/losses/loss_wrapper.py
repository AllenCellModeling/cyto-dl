import torch
from numpy.typing import ArrayLike
from torch import nn


class LossWrapper(nn.Module):
    def __init__(self, loss_fn, channel_weight: ArrayLike, loss_scale: float = 1.0):
        """Loss Wrapper for weighting loss between channels differently. `loss_fn` is calculated
        per-channel, scaled by `channel_weight`, averaged, and scaled by `loss_scale`

        Parameters
        ----------
            loss_fn
                Loss function
            channel_weight:ArrayLike
                array of floats with length equal to number of channels of predicted image
            loss_scale: float
                Scale for channel-weighted loss.
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.channel_weight = channel_weight
        self.loss_scale = loss_scale

    def __call__(self, y_hat, y):
        assert (
            len(self.channel_weight) == y.shape[1]
        ), "Channel size mismatch, please adjust channel weights."
        # calculate channel-wise loss, preserving NCZYX dims, and scale by channel weight
        loss = torch.stack(
            [
                torch.mul(
                    self.loss_fn(y_hat[:, i : i + 1], y[:, i : i + 1]),
                    self.channel_weight[i],
                )
                for i in range(len(self.channel_weight))
            ]
        ).mean()
        return loss * self.loss_scale


class CMAP_loss(nn.Module):
    def __init__(self, loss):
        """Loss Wrapper for losses that accept a spatial costmap, differentially emphasizing pixel
        losses throughout an image.

        Parameters
        ----------
            loss
                Loss function. Should provide per-pixel losses.
        """
        super().__init__()
        self.loss = loss

    def __call__(self, y_hat, y, cmap=None):
        self.loss = self.loss.to(y_hat.device)
        if cmap is None:
            return torch.mean(self.loss(y_hat, y.half()))

        # 2d head
        if len(y_hat.shape) == 4 and len(cmap.shape) == 5:
            cmap, _ = torch.max(cmap, dim=2)
        return torch.mean(torch.mul(self.loss(y_hat, y.half()), cmap))
