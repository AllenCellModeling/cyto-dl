import torch
from torch import nn

class LossWrapper(nn.Module):
    def __init__(self, loss_fn, channel_weight, loss_scale):
        self.name = loss_fn.__name__
        self.loss_fn = loss_fn(reduction="none")
        self.channel_weight = torch.Tensor(channel_weight).reshape(
            len(channel_weight), 1, 1, 1
        )
        self.loss_scale = loss_scale

    def per_item_loss(self, y_hat_item, y, channel_weight):
        return torch.mul(self.loss_fn(y_hat_item, y), channel_weight).mean()

    def __call__(self, y_hat, y):
        channel_weight = self.channel_weight.type_as(y)
        loss = 0
        for y_hat_item in y_hat:
            loss += self.per_item_loss(y_hat_item, y, channel_weight)
        return loss * self.loss_scale
