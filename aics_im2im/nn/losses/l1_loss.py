import torch


class L1Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.L1Loss(reduction="none")

    def forward(self, preds, gts):
        return self.loss(preds, gts).sum(-1).mean()
