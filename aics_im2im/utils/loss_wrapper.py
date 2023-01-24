import torch
from torch import nn


class LossWrapper(nn.Module):
    def __init__(self, loss_fn, channel_weight, loss_scale):
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
