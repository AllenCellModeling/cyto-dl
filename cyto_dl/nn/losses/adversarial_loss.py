from torch.nn.modules.loss import _Loss as Loss


class AdversarialLoss(Loss):
    def __init__(self, discriminator, loss, argmax=False, reduction="mean", squeeze=False):
        super().__init__(None, None, reduction)
        self.discriminator = discriminator
        self.loss = loss
        self.argmax = argmax
        self.squeeze = squeeze

    def forward(self, input, target, return_pred=False):
        yhat = self.discriminator(input)

        if self.squeeze:
            loss = self.loss(yhat, target.squeeze())
        elif self.argmax:
            loss = self.loss(yhat, target.argmax(1))
        else:
            loss = self.loss(yhat, target)

        # reduce across batch dimension
        if self.reduction == "none":
            pass
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        else:
            raise NotImplementedError(f"Unavailable reduction type: {self.reduction}")

        if return_pred:
            return loss, yhat

        return loss
