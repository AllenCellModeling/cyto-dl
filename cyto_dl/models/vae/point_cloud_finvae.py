import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from cyto_dl.nn.mlp import MLP

from cyto_dl.models.vae.point_cloud_vae import PointCloudVAE
from cyto_dl.models.vae.priors import IdentityPrior, IsotropicGaussianPrior
from cyto_dl.nn.losses import ChamferLoss
from torchmetrics import MeanMetric
from torch import distributions as dist

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
from torch.autograd import grad

logger = logging.getLogger("lightning")
logger.propagate = False


class PointCloudFinVAE(PointCloudVAE):
    def __init__(
        self,
        latent_dim: int,
        latent_dim_inv: int,
        latent_dim_spur: int,
        x_label: str,
        inv_covar_dim: int,
        spur_covar_dim: int,
        encoder: dict,
        decoder: dict,
        condition_keys: list,
        spur_keys: list,
        inv_keys: list,
        reconstruction_loss: dict,
        prior: dict,
        get_rotation: bool = False,
        beta: float = 1.0,
        reg_sm: float = 0,
        disable_metrics: bool = True,
        normalize_constant: float = 1,
        id_label: Optional[str] = None,
        point_label: Optional[str] = "points",
        occupancy_label: Optional[str] = "points.df",
        embedding_head: Optional[dict] = None,
        embedding_head_loss: Optional[dict] = None,
        embedding_head_weight: Optional[dict] = None,
        condition_encoder: Optional[dict] = None,
        condition_decoder: Optional[dict] = None,
        tc_beta: Optional[int] = None,
        kl_rate: Optional[float] = None,
        elbo_version: Optional[str] = None,
        inject_covar_in_latent: Optional[bool] = None,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        **base_kwargs,
    ):
        super().__init__(
            latent_dim=latent_dim,
            x_label=x_label,
            get_rotation=get_rotation,
            id_label=id_label,
            optimizer=optimizer,
            beta=beta,
            point_label=point_label,
            occupancy_label=occupancy_label,
            encoder=encoder,
            decoder=decoder,
            embedding_head=embedding_head,
            embedding_head_loss=embedding_head_loss,
            embedding_head_weight=embedding_head_weight,
            condition_encoder=condition_encoder,
            condition_decoder=condition_decoder,
            condition_keys=condition_keys,
            reconstruction_loss=reconstruction_loss,
            prior=prior,
            disable_metrics=disable_metrics,
        )
        self.tc_beta = tc_beta
        self.kl_rate = kl_rate
        self.elbo_version = elbo_version
        self.inject_covar_in_latent = inject_covar_in_latent
        self._training_hps = [self.beta, self.tc_beta]
        self.spur_keys = spur_keys
        self.inv_keys = inv_keys
        self.spur_covar_dim = spur_covar_dim
        self.inv_covar_dim = inv_covar_dim
        self.latent_dim_inv = latent_dim_inv
        self.latent_dim_spur = latent_dim_spur
        self.reg_sm = reg_sm
        self.normalize_constant = normalize_constant
        self.automatic_optimization = False

        self.encoder_dist = Normal(device="cuda:0")
        self.prior_dist_inv = Normal(device="cuda:0")
        self.prior_dist_spur = Normal(device="cuda:0")
        self.decoder_dist_fct = Normal(device="cuda:0")

        output_dim_prior_nn = 2 * self.latent_dim
        n_layers_prior = 2

        hidden_dim_prior = 128
        self.prior_mean_inv = torch.zeros(1)
        self.prior_mean_spur = torch.zeros(1)
        self.logl_inv = MLP(
            *[self.inv_covar_dim, self.latent_dim_inv],
            hidden_layers=[hidden_dim_prior] * n_layers_prior,
        )

        self.logl_spur = MLP(
            *[self.spur_covar_dim, self.latent_dim_spur],
            hidden_layers=[hidden_dim_prior] * n_layers_prior,
        )

    def warm_up(self, iteration):
        if self.warm_up_iters > 0:
            beta = min(1, iteration / self.warm_up_iters) * self.beta
            tc_beta = min(1, iteration / self.warm_up_iters) * self.tc_beta
            self._training_hps = [beta, tc_beta]

    def compute_prior(self, inv_covar, spur_covar):
        logl_inv = (
            (self.logl_inv(inv_covar.float()).exp() + 1e-4)
            if (inv_covar is not None)
            else self.logl_inv
        )

        logl_spur = self.logl_spur(spur_covar).exp() + 1e-4

        return (
            self.prior_mean_inv.type_as(logl_spur),
            self.prior_mean_spur.type_as(logl_spur),
        ), (logl_inv, logl_spur)

    def get_inv(self, batch):
        for j, key in enumerate(self.inv_keys):
            if j == 0:
                inv_covar = torch.squeeze(batch[key], dim=-1)
            else:
                inv_covar = torch.cat(
                    (inv_covar, torch.squeeze(batch[key], dim=-1)), dim=1
                )
        return inv_covar

    def get_spur(self, batch):
        for j, key in enumerate(self.spur_keys):
            if j == 0:
                spur_covar = torch.squeeze(batch[key], dim=-1)
            else:
                spur_covar = torch.cat(
                    (spur_covar, torch.squeeze(batch[key], dim=-1)), dim=1
                )
        return spur_covar

    def calculate_elbo(
        self, batch, xhat, z_params, z, stage
    ):  # z_params is unsampled, z is sampled with reparametrization trick
        log_px_z = -100 * self.reconstruction_loss[self.hparams.x_label](
            batch[self.hparams.x_label], xhat[self.hparams.x_label]
        )

        # log_px_z = log_normal(batch[self.hparams.x_label], xhat[self.hparams.x_label], self.decoder_var)

        if stage != "train":
            return log_px_z, z_params

        z = z[self.hparams.x_label]
        g = z_params["latent_mean"]
        v = z_params["latent_logvar"]

        beta, tc_beta = self._training_hps

        prior_params_mean = z_params["prior_params_mean"]
        prior_params_var = z_params["prior_params_var"]

        log_qz_xde = self.encoder_dist.log_pdf(z, g, v)
        log_pzi_d = self.prior_dist_inv.log_pdf(
            z[:, : self.latent_dim_inv], prior_params_mean[0], prior_params_var[0]
        )

        log_pzs_e = self.prior_dist_spur.log_pdf(
            z[:, self.latent_dim_inv :], prior_params_mean[1], prior_params_var[1]
        )

        return -(log_px_z + beta * (log_pzi_d + log_pzs_e - log_qz_xde)).mean(), z

    def sample_z(self, z_parts_params, inference=False):
        z_parts_params[self.hparams.x_label] = self.reparameterize(
            z_parts_params["latent_mean"], z_parts_params["latent_logvar"]
        )
        return z_parts_params

    def forward(self, batch, decode=False, inference=True, return_params=False):
        inv_covar = self.get_inv(batch)
        spur_covar = self.get_spur(batch)
        prior_params_mean, prior_params_var = self.compute_prior(inv_covar, spur_covar)
        is_inference = inference or not self.training

        z_params = self.encode(batch, get_rotation=self.get_rotation)
        z_params = self.encoder_compose_function(z_params)
        z_params["latent_logvar"] = z_params["latent_logvar"].exp() + 1e-4
        encoder_params = z_params["latent_mean"], z_params["latent_logvar"]
        sample_z = self.encoder_dist.sample(*encoder_params)
        z = {self.hparams.x_label: sample_z}
        z = self.decoder_compose_function(z, batch)
        z_params["prior_params_mean"] = prior_params_mean
        z_params["prior_params_var"] = prior_params_var

        if not decode:
            return z

        if hasattr(self.encoder[self.hparams.x_label], "generate_grid_feats"):
            if self.encoder[self.hparams.x_label].generate_grid_feats:
                xhat = self.decode(z, batch=batch)
            else:
                xhat = self.decode(z)
        else:
            xhat = self.decode(z)

        if return_params:
            return xhat, z, z_params

        return xhat, z

    def model_step(self, stage, batch, batch_idx):
        (
            xhat,
            z,
            z_params,
        ) = self.forward(batch, decode=True, inference=False, return_params=True)

        (loss, z) = self.calculate_elbo(batch, xhat, z_params, z, stage)

        loss = {
            "loss": loss,
        }

        preds = {}

        return loss, preds, None

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)
        loss, preds, targets = self.model_step("train", batch, batch_idx)
        self.manual_backward(loss["loss"])
        opt.step()
        self.compute_metrics(loss, preds, targets, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step("val", batch, batch_idx)
        self.compute_metrics(loss, preds, targets, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step("test", batch, batch_idx)
        self.compute_metrics(loss, preds, targets, "test")
        return loss

    def predict_step(self, batch, batch_idx):
        """Here you should implement the logic for an inference step.

        In most cases this would simply consist of calling the forward pass of your model, but you
        might wish to add additional post-processing.
        """
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "frequency": 1,
                },
            }
        return optimizer


def log_nb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
    log_fn: callable = torch.log,
    lgamma_fn: callable = torch.lgamma,
    reduce: bool = True,
):
    """
    Log likelihood (scalar) of a minibatch according to a nb model.
    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x genes)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
    eps
        numerical stability constant
    """

    log = log_fn
    lgamma = lgamma_fn

    log_theta_mu_eps = log(theta + mu + eps)

    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )

    if reduce:
        return res.sum(dim=-1)
    else:
        return res


def log_normal(x, mu=None, v=None, reduce=True):
    """Compute the log-pdf of a normal distribution with diagonal covariance"""
    # if mu.shape[1] != v.shape[0] and mu.shape != v.shape:
    #    raise ValueError(f'The mean and variance vector do not have the same shape:\n\tmean: {mu.shape}\tvariance: {v.shape}')

    logpdf = -0.5 * (np.log(2 * np.pi) + v.log() + (x - mu).pow(2).div(v))

    if reduce:
        logpdf = logpdf.sum(dim=-1)
        if len(logpdf.shape) > 1:
            logpdf = logpdf.sum(dim=-1)
        return logpdf

    return logpdf


class Dist:
    def __init__(self):
        pass

    def sample(self, *args):
        pass

    def log_pdf(self, *args, **kwargs):
        pass


class Normal(Dist):
    def __init__(self, device="cpu"):
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
        """compute the log-pdf of a normal distribution with diagonal covariance"""
        if param_shape is not None:
            mu, v = mu.view(param_shape), v.view(param_shape)
        lpdf = -0.5 * (torch.log(self.c) + v.log() + (x - mu).pow(2).div(v))
        if reduce:
            lpdf = lpdf.sum(dim=-1)
            if len(lpdf.shape) > 1:
                lpdf = lpdf.sum(dim=-1)
            return lpdf
        else:
            return lpdf
