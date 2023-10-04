import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from cyto_dl.models.vae.conditional_canon_vae import ConditionalCanonVAE
from cyto_dl.models.vae.priors import IdentityPrior, IsotropicGaussianPrior
from cyto_dl.nn.losses import ChamferLoss
from cyto_dl.nn.point_cloud import DGCNN, FoldingNet
from cyto_dl.nn.gaussian_mlp import Normal

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


class iVAE(ConditionalCanonVAE):
    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        x_label: str,
        encoder: dict,
        decoder: dict,
        reconstruction_loss: dict,
        prior: dict,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        get_rotation=False,
        beta: float = 1.0,
        id_label: Optional[str] = None,
        embedding_head: Optional[dict] = None,
        embedding_head_loss: Optional[dict] = None,
        embedding_head_weight: Optional[dict] = None,
        basal_head: Optional[dict] = None,
        basal_head_loss: Optional[dict] = None,
        basal_head_weight: Optional[dict] = None,
        condition_encoder: Optional[dict] = None,
        condition_decoder: Optional[dict] = None,
        condition_keys: Optional[list] = None,
        disable_metrics: Optional[bool] = False,
        **base_kwargs,
    ):
        metric_keys = [
            "train/loss",
            "val/loss",
            "test/loss",
            "train/loss/total_reconstruction",
            "val/loss/total_reconstruction",
            "test/loss/total_reconstruction",
        ]

        super().__init__(
            latent_dim=latent_dim,
            x_label=x_label,
            encoder=encoder,
            decoder=decoder,
            reconstruction_loss=reconstruction_loss,
            prior=prior,
            optimizer=optimizer,
            get_rotation=get_rotation,
            beta=beta,
            id_label=id_label,
            embedding_head=embedding_head,
            embedding_head_loss=embedding_head_loss,
            embedding_head_weight=embedding_head_weight,
            basal_head=basal_head,
            basal_head_loss=basal_head_loss,
            basal_head_weight=basal_head_weight,
            condition_encoder=condition_encoder,
            condition_decoder=condition_decoder,
            condition_keys=condition_keys,
            disable_metrics=disable_metrics,
            metric_keys=metric_keys
        )
        self.cond_dim = cond_dim
        self.occupancy_label = None
        self.l1 = nn.Linear(cond_dim, 200)
        self.l21 = nn.Linear(200, latent_dim)
        self.l22 = nn.Linear(200, latent_dim)

        self.encoder_dist = Normal()
        self.prior_dist = Normal()
        self._training_hyperparams = [1., 1., 1., 1., 1]
        self.anneal_params = False

    def prior_c(self, y):
        h2 = F.relu(self.l1(y))
        # h2 = self.l1(y)
        return self.l21(h2), self.l22(h2)

    def anneal(self, N, max_iter, it):
        thr = int(max_iter / 1.6)
        # a = 0.5 / self.decoder.log_var(0).exp().item()
        a = 0.5
        self._training_hyperparams[-1] = N
        self._training_hyperparams[0] = min(2 * a, a + a * it / thr)
        self._training_hyperparams[1] = max(1, a * .3 * (1 - it / thr))
        self._training_hyperparams[2] = min(1, it / thr)
        self._training_hyperparams[3] = max(1, a * .5 * (1 - it / thr))
        if it > thr:
            self.anneal_params = False

    def calculate_elbo(self, x, xhat, z, z_params):
        rcl_reduced = self.calculate_rcl_dict(x, xhat, z)

        mean_logvar = z_params[self.hparams.x_label]
        mean, logvar = torch.split(mean_logvar, mean_logvar.shape[1] // 2, dim=1)

        rcl_reduced = self.calculate_rcl_dict(x, xhat, z)
        prior_params = z['mup'], z['log1p']
        # log_pz_u =self.prior_dist.log_pdf(z[self.hparams.x_label], *prior_params)
        log_qz_xu =self.encoder_dist.log_pdf(z[self.hparams.x_label], mean, logvar)
        log_pz_u =self.prior_dist.log_pdf(z[self.hparams.x_label], *prior_params)
        total_recon = -sum(rcl_reduced.values())


        if self.anneal_params:
            log_px_z = total_recon
            a, b, c, d, N = self._training_hyperparams
            M = z[self.hparams.x_label].size(0)
            log_qz_tmp = self.encoder_dist.log_pdf(z[self.hparams.x_label].view(M, 1, self.latent_dim), mean.view(1, M, self.latent_dim),
                                                   logvar.view(1, M, self.latent_dim), reduce=False)
            log_qz = torch.logsumexp(log_qz_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
            log_qz_i = (torch.logsumexp(log_qz_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

            total_loss = a * log_px_z + ( - b * (log_qz_xu - log_qz) - c * (log_qz - log_qz_i) - d * (
                    log_qz_i - log_pz_u)).mean()
        else:
            total_loss = total_recon + (log_pz_u - log_qz_xu).mean()

        return (
            -total_loss,
            total_recon,
            rcl_reduced,
            (log_pz_u - log_qz_xu).mean(),
            log_pz_u - log_qz_xu,
        )

    def forward(self, batch, decode=False, inference=True, return_params=False):
        is_inference = inference or not self.training

        z_params = self.encode(batch, get_rotation=self.get_rotation)
        z_params = self.encoder_compose_function(z_params)

        z = self.sample_z(z_params, inference=inference)

        for key in self.condition_keys:
            mup, log1p = self.prior_c(batch[key])
            z['mup'] = mup
            z['log1p'] = log1p

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

        (
            loss,
            rec_loss,
            rec_loss_per_part,
            kld_loss,
            kld_per_part,
        ) = self.calculate_elbo(batch, xhat, z, z_params)

        loss = {
            "loss": loss,
            "total_kld": kld_loss.detach(),
            "total_reconstruction": rec_loss.detach(),
        }

        for part, recon_part in rec_loss_per_part.items():
            loss[f"reconstruction_{part}"] = recon_part.detach()

        return loss, None, None
