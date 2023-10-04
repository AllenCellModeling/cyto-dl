import numpy as np
import torch
import torch.nn as nn
from cyto_dl.nn.mlp import MLP


class FactorizedPrior(nn.Module):
    def __init__(
        self,
        latent_dim_spur,
        spur_covar_dim,
        hidden_dim_prior,
        n_layers_prior,
        fix_mean_spur_prior,
        fix_var_spur_prior,
        inject_covar_in_latent,
    ):
        super().__init__()

        self.fix_mean_spur_prior = fix_mean_spur_prior
        self.fix_var_spur_prior = fix_var_spur_prior
        self.spur_covar_dim = spur_covar_dim
        self.inject_covar_in_latent = inject_covar_in_latent

        if self.fix_mean_spur_prior and self.fix_var_spur_prior:
            self.prior_mean_spur = torch.zeros(1)
            self.logl_spur = torch.ones(1)
        elif self.fix_mean_spur_prior:
            self.prior_mean_spur = torch.zeros(1)
            if spur_covar_dim != 0:
                self.logl_spur = MLP(
                    *[self.spur_covar_dim, latent_dim_spur],
                    hidden_layers=[hidden_dim_prior] * n_layers_prior,
                )
            else:
                self.logl_spur = torch.ones(1)
        elif self.fix_var_spur_prior:
            self.logl_spur = torch.ones(1)
            if spur_covar_dim != 0:
                self.prior_mean_spur = MLP(
                    *[self.spur_covar_dim, latent_dim_spur],
                    hidden_layers=[hidden_dim_prior] * n_layers_prior,
                )
            else:
                self.prior_mean_spur = torch.zeros(1)
        elif (not self.fix_mean_spur_prior) and (not self.fix_var_spur_prior):
            ## Noise latent prior
            if spur_covar_dim != 0:
                self.prior_nn_spur = MLP(
                    *[self.spur_covar_dim, hidden_dim_prior],
                    hidden_layers=[hidden_dim_prior] * n_layers_prior,
                )

                self.prior_mean_spur = nn.Linear(hidden_dim_prior, latent_dim_spur)
                self.logl_spur = nn.Linear(hidden_dim_prior, latent_dim_spur)
            else:
                self.prior_mean_spur = torch.zeros(1)
                self.logl_spur = torch.ones(1)

    def forward(self, spur_covar, **kwargs):
        if self.fix_mean_spur_prior and self.fix_var_spur_prior:
            return self.prior_mean_spur, self.logl_spur
        elif self.fix_mean_spur_prior:
            if (spur_covar is not None) and (not self.inject_covar_in_latent):
                logl_spur = self.logl_spur(spur_covar).exp() + 1e-4
            else:
                logl_spur = self.logl_spur

            return self.prior_mean_spur, logl_spur
        elif self.fix_var_spur_prior:
            if (spur_covar is not None) and (not self.inject_covar_in_latent):
                prior_mean_spur = self.prior_mean_spur(spur_covar)
            else:
                prior_mean_spur = self.prior_mean_spur

            return prior_mean_spur, self.logl_spur
        elif (not self.fix_mean_spur_prior) and (not self.fix_var_spur_prior):
            if (spur_covar is not None) and (not self.inject_covar_in_latent):
                prior_shared_spur = self.prior_nn_spur(spur_covar)

                prior_mean_spur = self.prior_mean_spur(prior_shared_spur)
                logl_spur = self.logl_spur(prior_shared_spur).exp() + 1e-4
            else:
                prior_mean_spur = self.prior_mean_spur
                logl_spur = self.logl_spur

            return prior_mean_spur, logl_spur
