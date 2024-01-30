import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
import math
from cyto_dl.models.vae.point_cloud_nfinvae import PointCloudNFinVAE
from cyto_dl.models.vae.priors import IdentityPrior, IsotropicGaussianPrior
from cyto_dl.nn.losses import ChamferLoss
import cyto_dl
from torchmetrics import MeanMetric

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
from torch.autograd import grad

logger = logging.getLogger("lightning")
logger.propagate = False


class PointCloudNFinVAEMultarget(PointCloudNFinVAE):
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
        **base_kwargs,
    ):
        super().__init__(
            latent_dim=latent_dim,
            latent_dim_inv=latent_dim_inv,
            latent_dim_spur=latent_dim_spur,
            x_label=x_label,
            encoder=encoder,
            decoder=decoder,
            condition_keys=condition_keys,
            spur_keys=spur_keys,
            inv_keys=inv_keys,
            reconstruction_loss=reconstruction_loss,
            prior=prior,
            get_rotation=get_rotation,
            beta=beta,
            reg_sm=reg_sm,
            disable_metrics=disable_metrics,
            normalize_constant=normalize_constant,
            id_labe=id_label,
            point_labe=point_label,
            occupancy_label=occupancy_label,
            embedding_head=embedding_head,
            embedding_head_loss=embedding_head_loss,
            embedding_head_weight=embedding_head_weight,
            condition_encoder=condition_encoder,
            condition_decoder=condition_decoder,
            tc_beta=tc_beta,
            kl_rate=kl_rate,
            elbo_version=elbo_version,
            condition_decoder_keys=condition_decoder_keys,
            inject_covar_in_latent=inject_covar_in_latent,
            one_hot_dict=one_hot_dict,
            dataset_size=dataset_size,
            optimizer=optimizer,
            target_key=target_key,
            x_dim=x_dim,
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

    def parse_batch(self, batch):
        if self.one_hot_dict:
            for key in self.one_hot_dict.keys():
                batch[key] = torch.nn.functional.one_hot(batch[key].long(), num_classes = self.one_hot_dict[key]['num_classes']).float()

        return batch
    
    def encode(self, batch, **kwargs):
        ret_dict = {}
        for part, encoder in self.encoder.items():
            this_batch_part = batch[part]
            this_ret = encoder(
                this_batch_part,
                **{k: v for k, v in kwargs.items() if k in self.encoder_args[part]},
            )
            if isinstance(this_ret, dict):  # deal with multiple outputs for an encoder
                for key in this_ret.keys():
                    new_key = key
                    if key == 'rotation':
                        new_key = part + '_rotation'
                    ret_dict[new_key] = this_ret[key]
            else:
                ret_dict[part] = this_ret
        return ret_dict
    
    def decode(self, z_parts, return_canonical=False, batch=None):
        base_embed = z_parts[self.hparams.x_label]
        batch_size = base_embed.shape[0]
        xhat_dict = {}
        for j, key in enumerate(self.target_key):
            # this_ind = torch.ones([batch_size,1]).fill_(j).long().type_as(base_embed)
            # cond_feats = torch.cat(
            #     (this_ind, z_parts[self.hparams.x_label]), dim=1
            # )  
            # z_parts[key] = self.condition_decoder[
            #     self.hparams.x_label
            # ](cond_feats)

            base_xhat = self.decoder[key](
                base_embed
            )

            if self.get_rotation:
                rotation = z_parts[key + "_rotation"]
                xhat = torch.einsum("bij,bjk->bik", base_xhat[:, :, :3], rotation)
                if xhat.shape[-1] != base_xhat.shape[-1]:
                    xhat = torch.cat([xhat, base_xhat[:, :, -1:]], dim=-1)
            else:
                xhat = base_xhat
            xhat_dict[key] = xhat
            if return_canonical:
                xhat_dict[key + '_canonical'] = base_xhat
            
        return xhat_dict

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
    
    def calculate_rcl(self, batch, xhat):
        for j, key in enumerate(self.target_key):
            this_log_px_z = -0.1 * self.reconstruction_loss[self.hparams.x_label](
                batch[key], xhat[key]
            )
            if j == 0:
                log_px_z = this_log_px_z
            else:
                log_px_z = log_px_z + this_log_px_z

        return log_px_z