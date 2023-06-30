import torch

from .abstract_prior import Prior


class IdentityPrior(Prior):
    """Prior class that doesn't contribute to KL loss.

    Effectively a Dirac delta distribution given z.
    """

    def forward(self, z, mode="kl", **kwargs):
        if mode == "kl":
            return torch.tensor(0.0).type_as(z)
        return z

    @property
    def param_size(self):
        return self.dimensionality
