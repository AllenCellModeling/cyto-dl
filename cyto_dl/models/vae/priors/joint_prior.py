import torch
import torch.nn as nn

from .abstract_prior import Prior


class JointPrior(Prior):
    def __init__(self, priors):
        dimensionality = sum(prior.dimensionality for prior in priors)
        super().__init__(dimensionality)

        if not isinstance(priors, nn.ModuleList):
            priors = nn.ModuleList(*priors)
        self.priors = priors

    @property
    def param_size(self):
        return sum(_.param_size for _ in self.priors)

    def kl_divergence(self, z_params, reduction="sum"):
        kl = []

        start_ix = 0
        for prior in self.priors:
            end_ix = (start_ix + prior.dimensionality) - 1
            kl.append(prior(z_params[:, start_ix:end_ix], mode="kl", reduction=reduction))
            start_ix = end_ix + 1

        kl = torch.cat(kl, axis=1)

        if reduction == "none":
            return kl
        elif reduction == "sum":
            return kl.sum(dim=-1)
        else:
            raise NotImplementedError(f"Reduction '{reduction}' not implemented for JointPrior")

    def sample(self, z_params, inference=False):
        samples = []

        start_ix = 0
        for prior in self.priors:
            end_ix = (start_ix + prior.dimensionality) - 1
            samples.append(prior(z_params[:, start_ix:end_ix], mode="sample", inference=inference))
            start_ix = end_ix + 1

        return torch.cat(samples, axis=1)

    def forward(self, z_params, mode="kl", inference=False, **kwargs):
        if mode == "kl":
            return self.kl_divergence(z_params)

        return self.sample(z_params, inference=inference)
