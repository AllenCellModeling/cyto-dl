"""Adapted from.

- https://github.com/siamakz/iVAE/tree/master
"""

from typing import Optional

import numpy as np
import torch
from torch import distributions as dist
from torch import nn

from .mlp import MLP


class Dist:
    def __init__(self):
        pass

    def sample(self, *args):
        pass

    def log_pdf(self, *args, **kwargs):
        pass


class Normal(Dist):
    def __init__(self, device="cuda:0"):
        super().__init__()
        self.device = device
        self.c = 2 * np.pi * torch.ones(1).to(self.device)
        self._dist = dist.normal.Normal(
            torch.zeros(1).to(self.device), torch.ones(1).to(self.device)
        )
        self.name = "gauss"

    def sample(self, mu, v):
        eps = self._dist.sample(mu.size()).squeeze()
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def log_pdf(self, x, mu, v, reduce=True, param_shape=None):
        """compute the log-pdf of a normal distribution with diagonal covariance."""
        if param_shape is not None:
            mu, v = mu.view(param_shape), v.view(param_shape)

        lpdf = -0.5 * (torch.log(self.c) + torch.nan_to_num(v.log()) + (x - mu).pow(2).div(v))
        if reduce:
            return lpdf.sum(dim=-1)
        return lpdf

    def log_pdf_full(self, x, mu, v):
        """compute the log-pdf of a normal distribution with full covariance v is a batch of
        "pseudo sqrt" of covariance matrices of shape (batch_size, d_latent, d_latent) mu is batch
        of means of shape (batch_size, d_latent)"""
        batch_size, d = mu.size()
        cov = torch.einsum("bik,bjk->bij", v, v)  # compute batch cov from its "pseudo sqrt"
        assert cov.size() == (batch_size, d, d)
        inv_cov = torch.inverse(cov)  # works on batches
        c = d * torch.log(self.c)
        # matrix log det doesn't work on batches!
        _, logabsdets = self._batch_slogdet(cov)
        xmu = x - mu
        return -0.5 * (c + logabsdets + torch.einsum("bi,bij,bj->b", [xmu, inv_cov, xmu]))

    def _batch_slogdet(self, cov_batch: torch.Tensor):
        """compute the log of the absolute value of determinants for a batch of 2D matrices.

        Uses torch.slogdet this implementation is just a for loop, but that is what's suggested in
        torch forums gpu compatible
        """
        batch_size = cov_batch.size(0)
        signs = torch.empty(batch_size, requires_grad=False)
        logabsdets = torch.empty(batch_size, requires_grad=False)
        for i, cov in enumerate(cov_batch):
            signs[i], logabsdets[i] = torch.slogdet(cov)
        return signs, logabsdets


class GaussianMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        n_layers,
        activation=None,
        slope=None,
        fixed_mean=None,
        fixed_var=None,
    ):
        super().__init__()
        self.distribution = Normal()
        if fixed_mean is None:
            self.mean = MLP(*[input_dim, output_dim], hidden_layers=[hidden_dim] * n_layers)
        else:
            self.mean = lambda x: fixed_mean * torch.ones(1)
        if fixed_var is None:
            self.log_var = MLP(*[input_dim, output_dim], hidden_layers=[hidden_dim] * n_layers)
        else:
            self.log_var = lambda x: np.log(fixed_var) * torch.ones(1)

    def sample(self, *params):
        return self.distribution.sample(*params)

    def log_pdf(self, x, *params, **kwargs):
        return self.distribution.log_pdf(x, *params, **kwargs)

    def forward(self, *input):
        if len(input) > 1:
            x = torch.cat(input, dim=1)
        else:
            x = input[0]
        return self.mean(x), self.log_var(x).exp()
