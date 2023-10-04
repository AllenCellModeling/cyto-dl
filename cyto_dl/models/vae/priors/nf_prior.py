import numpy as np
import torch
import torch.nn as nn
from cyto_dl.nn.mlp import MLP


class NonFactorizedPrior(nn.Module):
    def __init__(
        self,
        latent_dim_inv,
        inv_covar_dim,
        output_dim_prior_nn,
        hidden_dim_prior,
        n_layers_prior,
    ):
        super().__init__()
        self.latent_dim_inv = latent_dim_inv
        self.output_dim_prior_nn = output_dim_prior_nn
        self.inv_covar_dim = inv_covar_dim
        self.t_nn = MLP(
            *[self.latent_dim_inv, output_dim_prior_nn],
            hidden_layers=[hidden_dim_prior] * n_layers_prior,
        )

        self.params_t_nn = MLP(
            *[self.inv_covar_dim, output_dim_prior_nn],
            hidden_layers=[hidden_dim_prior] * n_layers_prior,
        )

        self.params_t_suff = MLP(
            *[self.inv_covar_dim, 2 * latent_dim_inv],
            hidden_layers=[hidden_dim_prior] * n_layers_prior,
        )

    def forward(self, z, inv_covar):
        t_nn = self.t_nn(z)
        params_t_nn = self.params_t_nn(inv_covar.float())

        t_suff = torch.cat((z, z**2), dim=1)
        params_t_suff = self.params_t_suff(inv_covar.float())

        return t_nn, params_t_nn, t_suff, params_t_suff
