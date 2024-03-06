import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
import math
from cyto_dl.models.vae.point_cloud_vae import PointCloudVAE
from cyto_dl.models.vae.priors import IdentityPrior, IsotropicGaussianPrior
from cyto_dl.nn.losses import ChamferLoss
import cyto_dl
from torchmetrics import MeanMetric

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
from torch.autograd import grad
from cyto_dl.nn.losses import TopoLoss

logger = logging.getLogger("lightning")
logger.propagate = False


class PointCloudNFinVAE(PointCloudVAE):
    def __init__(
        self,
        latent_dim: int,
        latent_dim_inv: int,
        latent_dim_spur: int,
        x_label: str,
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
        tc_beta: Optional[int] = 1,
        kl_rate: Optional[float] = None,
        elbo_version: Optional[str] = None,
        condition_decoder_keys: Optional[list] = None,
        inject_covar_in_latent: Optional[bool] = None,
        one_hot_dict: Optional[dict] = None,
        dataset_size: Optional[int] = None,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        target_key: Optional[list] = None,
        x_dim: Optional[int] = None,
        include_top_loss: Optional[bool] = False,
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
        self.target_key = target_key
        self.kl_rate = kl_rate
        self.elbo_version = elbo_version
        self.inject_covar_in_latent = inject_covar_in_latent
        self._training_hps = [self.beta, self.tc_beta]
        self.spur_keys = spur_keys
        self.inv_keys = inv_keys
        self.latent_dim_inv = latent_dim_inv
        self.latent_dim_spur = latent_dim_spur
        self.reg_sm = reg_sm
        self.normalize_constant = normalize_constant
        self.automatic_optimization = False
        if x_dim is None:
            x_dim = 128
        self.decoder_var = torch.mul(0.01, torch.ones(x_dim))
        self.one_hot_dict = one_hot_dict
        self.dataset_size = dataset_size
        self.condition_decoder_keys = condition_decoder_keys
        self.include_top_loss = include_top_loss

        if self.include_top_loss:
            # self.top_loss = nn.ModuleDict({x_label: TopoLoss(topo_lambda=0.001, farthest_point=True, num_groups=50, mean=False)})
            self.top_loss = nn.ModuleDict({x_label: TopoLoss(topo_lambda=0.001, farthest_point=True, num_groups=256, mean=False)})

    def warm_up(self, iteration):
        if self.warm_up_iters > 0:
            beta = min(1, iteration / self.warm_up_iters) * self.beta
            tc_beta = min(1, iteration / self.warm_up_iters) * self.tc_beta
            # self._training_hps = [beta, tc_beta]
            # self._training_hps = [1, 0]

    def reparameterize(self, mean, logvar):
        std = (torch.exp(logvar) + 1e-4).sqrt()
        eps = torch.randn_like(std)
        return mean + eps * std

    def parse_batch(self, batch):
        if self.one_hot_dict:
            for key in self.one_hot_dict.keys():
                batch[key] = torch.nn.functional.one_hot(batch[key].long(), num_classes = self.one_hot_dict[key]['num_classes']).float()

        if self.target_key is not None:
            for j, key in enumerate(self.target_key):                
                if j == 0:
                    batch['target'] = batch[f"{key}"]
                else:
                    batch['target'] = torch.cat([batch['target'], batch[f"{key}"]], dim=1)
            self.target_label = 'target'
        else:
            self.target_label = self.hparams.x_label
        return batch

    def decoder_compose_function(self, z_parts, batch):
        if (self.condition_keys is not None) & (self.condition_decoder_keys is not None):
            for j, key in enumerate(self.condition_decoder_keys):
                if j == 0:
                    cond_inputs = batch[key]
                    # cond_inputs = torch.squeeze(batch[key], dim=(-1))
                else:
                    cond_inputs = torch.cat((cond_inputs, batch[key]), dim=1)
            cond_feats = torch.cat(
                (cond_inputs, z_parts[self.hparams.x_label]), dim=1
            )
            # shared decoder
            z_parts[self.hparams.x_label] = self.condition_decoder[
                self.hparams.x_label
            ](cond_feats)
        return z_parts
    
    def calculate_rcl(self, batch, xhat):

        log_px_z = -0.1 * self.reconstruction_loss[self.hparams.x_label](
            batch[self.hparams.x_label], xhat[self.hparams.x_label]
        )
        # log_px_z = -5 * self.reconstruction_loss[self.hparams.x_label](
        #     batch[self.hparams.x_label], xhat[self.hparams.x_label]
        # )
        if self.include_top_loss:
            this_top_loss = -self.top_loss[self.hparams.x_label](batch[self.hparams.x_label], xhat[self.hparams.x_label])
        # log_px_z = log_normal(batch[self.target_label], xhat[self.hparams.x_label], self.decoder_var)
            return log_px_z + this_top_loss
        else:
            return log_px_z
        
    @torch.enable_grad()
    def calculate_elbo(
        self, batch, xhat, z, z_params, stage
    ):  # z_params is unsampled, z is sampled with reparametrization trick
        self.decoder_var = self.decoder_var.type_as(xhat[self.hparams.x_label])
        # import ipdb
        # ipdb.set_trace()
        log_px_z = self.calculate_rcl(batch, xhat)

        # if stage != "train":
        #     return log_px_z, z_params

        x = batch[self.hparams.x_label]
        device = x.device
        batch_size = x.shape[0]
        z = z[self.hparams.x_label]
        dataset_size = self.dataset_size

        for j, key in enumerate(self.inv_keys):
            if j == 0:
                inv_covar = z_params[key].view(batch_size, -1)
            else:
                inv_covar = torch.cat((inv_covar, z_params[key]), dim=1)
        inv_covar = inv_covar.long()
        # import ipdb
        # ipdb.set_trace()

        for j, key in enumerate(self.spur_keys):
            if j == 0:
                spur_covar = z_params[key].view(batch_size, -1)
            else:
                spur_covar = torch.cat((spur_covar, z_params[key]), dim=1)
        spur_covar = spur_covar.long()

        beta, tc_beta = self._training_hps

        if tc_beta > 0 and dataset_size is None:
            raise ValueError(
                "Dataset_size not given to elbo function, can not calculate Total Correlation loss part!"
            )

        # Warm-up for kl term and tc loss

        latent_mean = z_params["latent_mean"]
        latent_logvar = z_params["latent_logvar"]

        log_qz_xde = log_normal(z, latent_mean, (latent_logvar.exp() + 1e-4))
        # log_qz_xde = log_qz_xde.clamp(-1000)


        if tc_beta > 0:
            _logqz = log_normal(
                z.view(batch_size, 1, self.latent_dim),
                latent_mean.view(1, batch_size, self.latent_dim),
                (latent_logvar.exp() + 1e-4).view(1, batch_size, self.latent_dim),
                reduce=False,
            )

            # minibatch weighted sampling
            # logqz_prodmarginals = (
            #     torch.logsumexp(_logqz, dim=1, keepdim=False)
            #     - math.log(batch_size * dataset_size)
            # ).sum(1)
            # logqz = torch.logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(
            #     batch_size * dataset_size
            # )
            logqz = (torch.logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
            logqz_inv = (torch.logsumexp(_logqz[:,:,:self.latent_dim_inv].sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
            logqz_spur = (torch.logsumexp(_logqz[:,:,self.latent_dim_inv:].sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
            # print(f'The tc loss is {(tc_beta * (logqz - logqz_inv - logqz_spur)).mean().item()}')


        # Clone z first then calculate parts of the prior with derivative wrt cloned z
        # prior
        z_inv_copy = z[:, : self.latent_dim_inv].detach().requires_grad_(True)

        # Only use the latent invariant space
        t_nn, params_t_nn, t_suff, params_t_suff = self.prior[self.hparams.x_label](
            z_inv_copy, inv_covar
        )
        output_dim_prior_nn = self.prior[self.hparams.x_label].output_dim_prior_nn

        self.t_nn = self.prior[self.hparams.x_label].t_nn
        self.params_t_nn = self.prior[self.hparams.x_label].params_t_nn
        self.params_t_suff = self.prior[self.hparams.x_label].params_t_suff

        # Batched dot product for unnormalized prior probability
        log_pz_d_inv = torch.bmm(
            t_nn.view((-1, 1, output_dim_prior_nn)),
            params_t_nn.view((-1, output_dim_prior_nn, 1)),
        ).view(-1) + torch.bmm(
            t_suff.view((-1, 1, self.latent_dim_inv * 2)),
            params_t_suff.view((-1, self.latent_dim_inv * 2, 1)),
        ).view(
            -1
        )

        # Implement constant log prior so prior params are not updated but grads are backpropagated for the encoder
        self.t_nn.requires_grad_(False)
        self.params_t_nn.requires_grad_(False)
        self.params_t_suff.requires_grad_(False)

        t_nn_copy = self.t_nn(z[:, : self.latent_dim_inv])
        params_t_nn_copy = self.params_t_nn(inv_covar.float())

        t_suff_copy = torch.cat(
            (z[:, : self.latent_dim_inv], (z[:, : self.latent_dim_inv]) ** 2), dim=1
        )
        params_t_suff_copy = self.params_t_suff(inv_covar.float())

        log_pz_d_inv_copy = torch.bmm(
            t_nn_copy.view((-1, 1, output_dim_prior_nn)),
            params_t_nn_copy.view((-1, output_dim_prior_nn, 1)),
        ).view(-1) + torch.bmm(
            t_suff_copy.view((-1, 1, self.latent_dim_inv * 2)),
            params_t_suff_copy.view((-1, self.latent_dim_inv * 2, 1)),
        ).view(
            -1
        )
        # log_pz_d_inv_copy = log_pz_d_inv_copy.clamp(-3)

        self.t_nn.requires_grad_(True)
        self.params_t_nn.requires_grad_(True)
        self.params_t_suff.requires_grad_(True)
        
        # Calculate derivatives of prior automatically
        dprior_dz = grad(
            log_pz_d_inv,
            z_inv_copy,
            grad_outputs=torch.ones(log_pz_d_inv.shape, device="cuda:0"),
            create_graph=True,
        )[0]
        d2prior_d2z = grad(
            dprior_dz,
            z_inv_copy,
            grad_outputs=torch.ones(dprior_dz.shape, device="cuda:0"),
            create_graph=True,
        )[0]

        # Spurious prior
        prior_mean_spur, prior_var_spur = self.prior["spurious"](spur_covar)
        prior_mean_spur = prior_mean_spur.type_as(z)
        prior_var_spur = prior_var_spur.type_as(z)

        if not self.inject_covar_in_latent:
            log_pz_e_spur = log_normal(
                z[:, self.latent_dim_inv :],
                prior_mean_spur
                * torch.ones(
                    prior_var_spur.shape, device=device
                ),  # need for shape match (could change check in log_normal function)
                prior_var_spur,
            )
        else:
            log_pz_e_spur = 0

        if self.reg_sm == 0:
            sm_part = (
                (d2prior_d2z + torch.mul(0.5, torch.pow(dprior_dz, 2)))
                .sum(dim=1)
                .mean()
            )
        else:
            sm_part = (
                (
                    d2prior_d2z
                    + torch.mul(0.5, torch.pow(dprior_dz, 2))
                    + d2prior_d2z.pow(2).mul(self.reg_sm)
                )
                .sum(dim=1)
                .mean()
            )
        # print(logqz - logqz_prodmarginals, log_pz_d_inv_copy + log_pz_e_spur - log_qz_xde, log_qz_xde)
        if tc_beta > 0:
            objective_function = (
                (
                    log_px_z + 
                    beta * (log_pz_d_inv_copy + log_pz_e_spur - log_qz_xde) -
                    tc_beta * (logqz - logqz_inv - logqz_spur)
                ).mean().div(self.normalize_constant) -
                beta * sm_part
            )
        else:
            objective_function = (
                log_px_z + beta * (log_pz_d_inv_copy + log_pz_e_spur - log_qz_xde)
            ).mean().div(self.normalize_constant) - beta * sm_part
        objective_function = objective_function.mul(-1)
        return objective_function, z

    def sample_z(self, z_parts_params, inference=False):
        if inference:
            z_parts_params[self.hparams.x_label] = z_parts_params['latent_mean']
        else:
            z_parts_params[self.hparams.x_label] = self.reparameterize(
                z_parts_params["latent_mean"], z_parts_params["latent_logvar"]
            )
        return z_parts_params

    def model_step(self, stage, batch, batch_idx):

        (
            xhat,
            z,
            z_params,
        ) = self.forward(batch, decode=True, inference=False, return_params=True)

        (loss, z) = self.calculate_elbo(batch, xhat, z, z_params, stage)

        loss = {
            "loss": loss,
        }

        preds = {}

        return loss, preds, None

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)
        self.warm_up_epochs = 2
        self.warm_up_iters = self.warm_up_epochs * self.dataset_size/batch[self.hparams.x_label].shape[0]
        self.warm_up(self.global_step)
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
