import numpy as np
import torch
import torch.nn as nn

from .abstract_prior import Prior


def compute_tc_penalty(logvar):
    return (2 * logvar).exp().mean(dim=0).sum()


class IsotropicGaussianPrior(Prior):
    def __init__(self, *, dimensionality, clamp_logvar=8, tc_penalty_weight=None):
        self.tc_penalty_weight = tc_penalty_weight
        self.clamp_logvar = float(clamp_logvar)
        super().__init__(dimensionality)

    @property
    def param_size(self):
        return 2 * self.dimensionality

    @classmethod
    def kl_divergence(cls, mean, logvar, tc_penalty_weight=None, reduction="sum"):
        """Computes the Kullback-Leibler divergence between a diagonal gaussian and an isotropic
        (0,1) gaussian. It also works batch-wise.

        Parameters
        ----------
        mean: torch.Tensor
            Mean of the gaussian (or batch of gaussians)

        logvar: torch.Tensor
            Log-variance of the gaussian (or batch of gaussians)
        """

        kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())

        if reduction == "none":
            loss = kl
        elif reduction == "mean":
            loss = kl.mean(dim=-1)
        else:
            loss = kl.sum(dim=-1)

        if tc_penalty_weight is not None and reduction != "none":
            tc_penalty = compute_tc_penalty(logvar)
            loss = loss + tc_penalty_weight * tc_penalty

        return loss

    @classmethod
    def sample(cls, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).mul(std).add(mean)
        return eps

    def forward(self, z, mode="kl", inference=False, **kwargs):
        mean_logvar = z
        mean, logvar = torch.split(mean_logvar, mean_logvar.shape[1] // 2, dim=1)

        if self.clamp_logvar:
            logvar = torch.clamp(logvar, max=abs(self.clamp_logvar))

        if mode == "kl":
            return self.kl_divergence(
                mean, logvar, tc_penalty_weight=self.tc_penalty_weight, **kwargs
            )

        if inference:
            return mean

        return self.sample(mean, logvar, **kwargs)


class DiagonalGaussianPrior(IsotropicGaussianPrior):
    def __init__(
        self,
        dimensionality=None,
        mean=None,
        logvar=None,
        learn_mean=False,
        learn_logvar=False,
        clamp_logvar=8.0,
        tc_penalty_weight=None,
    ):
        if hasattr(mean, "__len__"):
            if dimensionality is None:
                dimensionality = len(mean)
            else:
                assert dimensionality == len(mean)

        if hasattr(logvar, "__len__"):
            if dimensionality is None:
                dimensionality = len(logvar)
            else:
                assert dimensionality == len(logvar)

        assert dimensionality is not None

        super().__init__(clamp_logvar, dimensionality)

        if logvar is None:
            logvar = torch.zeros(self.dimensionality)
        else:
            if not hasattr(logvar, "__len__"):
                logvar = [logvar] * self.dimensionality
            logvar = torch.tensor(np.fromiter(logvar, dtype=float))

        if learn_logvar:
            logvar = nn.Parameter(logvar, requires_grad=True)

        self.logvar = logvar

        if mean is None:
            mean = torch.zeros(self.dimension)
        else:
            if not hasattr(mean, "__len__"):
                logvar = [mean] * self.dimensionality
            mean = torch.tensor(np.fromiter(mean, dtype=float))

        if learn_mean:
            mean = nn.Parameter(mean, requires_grad=True)

        self.mean = mean
        self.tc_penalty_weight = tc_penalty_weight

    @property
    def param_size(self):
        return 2 * self.dimensionality

    @classmethod
    def kl_divergence(cls, mu1, mu2, logvar1, logvar2, tc_penalty_weight=None, reduction="sum"):
        """Computes the Kullback-Leibler divergence between two diagonal gaussians (not necessarily
        isotropic). It also works batch-wise.

        Parameters
        ----------
        mu1: torch.Tensor
            Mean of the first gaussian (or batch of first gaussians)
        mu2: torch.Tensor
            Mean of the second gaussian (or batch of second gaussians)
        logvar1: torch.Tensor
            Log-variance of the first gaussian (or batch of first gaussians)
        logvar2: torch.Tensor
            Log-variance of the second gaussian (or batch of second gaussians)
        """
        mu_diff = mu2 - mu1

        kl = 0.5 * (
            (logvar2 - logvar1) + (logvar1 - logvar2).exp() + (mu_diff.pow(2) / logvar2.exp()) + -1
        )

        if reduction == "none":
            return kl

        loss = kl.sum(dim=-1).mean()

        if tc_penalty_weight is not None:
            tc_penalty = compute_tc_penalty(logvar1)
            loss = loss + tc_penalty_weight * tc_penalty

        return loss

    def forward(self, z, mode="kl", inference=False, **kwargs):
        mean_logvar = z
        mean, logvar = torch.split(mean_logvar, mean_logvar.shape[1] // 2, dim=1)

        if self.clamp_logvar:
            logvar = torch.clamp(logvar, max=abs(self.clamp_logvar))

        if mode == "kl":
            return self.kl_divergence(
                mean,
                self.mean,
                logvar,
                self.logvar,
                tc_penalty_weight=self.tc_penalty_weight,
                **kwargs,
            )

        if inference:
            return mean

        return self.sample(mean, logvar, **kwargs)
