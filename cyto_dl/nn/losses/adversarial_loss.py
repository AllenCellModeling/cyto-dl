from torch.nn.modules.loss import _Loss as Loss


class AdversarialLoss(Loss):
    def __init__(self, discriminator, loss, argmax=False, reduction="mean"):
        super().__init__(None, None, reduction)
        self.discriminator = discriminator
        self.loss = loss
        self.argmax = argmax

    def forward(self, input, target):
        yhat = self.discriminator(input)
        if not self.argmax:
            loss = self.loss(yhat, target)
        else:
            loss = self.loss(yhat, target.argmax(1))

        # reduce across batch dimension
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        else:
            raise NotImplementedError(f"Unavailable reduction type: {self.reduction}")
