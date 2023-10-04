import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from cyto_dl.models.vae.base_vae import BaseVAE
from cyto_dl.models.vae.priors import IdentityPrior, IsotropicGaussianPrior
from cyto_dl.nn.losses import ChamferLoss
from cyto_dl.nn.point_cloud import DGCNN, FoldingNet

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


class ConditionalCanonVAE(BaseVAE):
    def __init__(
        self,
        latent_dim: int,
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
        metric_keys: Optional[list] = None,
        **base_kwargs,
    ):
        self.get_rotation = get_rotation
        self.condition_keys = condition_keys
        self.embedding_head = embedding_head
        self.embedding_head_loss = embedding_head_loss
        self.embedding_head_weight = embedding_head_weight
        self.basal_head = basal_head
        self.basal_head_loss = basal_head_loss
        self.basal_head_weight = basal_head_weight
        self.disable_metrics = disable_metrics

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            x_label=x_label,
            id_label=id_label,
            beta=beta,
            reconstruction_loss=reconstruction_loss,
            optimizer=optimizer,
            prior=prior,
            disable_metrics=disable_metrics,
            metric_keys=metric_keys,
        )

        self.condition_encoder = nn.ModuleDict(condition_encoder)
        self.condition_decoder = nn.ModuleDict(condition_decoder)
        self.embedding_head = nn.ModuleDict(embedding_head)
        self.embedding_head_loss = nn.ModuleDict(embedding_head_loss)
        self.basal_head = nn.ModuleDict(basal_head)
        self.basal_head_loss = nn.ModuleDict(basal_head_loss)

    def encoder_compose_function(self, z_parts):
        if self.basal_head:
            z_parts[self.hparams.x_label + "_basal"] = z_parts[self.hparams.x_label]
            for key in self.basal_head.keys():
                z_parts[key] = self.basal_head[key](
                    z_parts[self.hparams.x_label + "_basal"]
                )

        if self.condition_keys:
            for j, key in enumerate([self.hparams.x_label] + self.condition_keys):
                this_z_parts = z_parts[key]
                if len(this_z_parts.shape) == 3:
                    this_z_parts = torch.squeeze(z_parts[key], dim=(-1))
                    z_parts[key] = this_z_parts
                    # this_z_parts = this_z_parts.argmax(dim=1)
                if j == 0:
                    cond_feats = this_z_parts
                else:
                    cond_feats = torch.cat((cond_feats, this_z_parts), dim=1)
            # shared encoder
            z_parts[self.hparams.x_label] = self.condition_encoder[
                self.hparams.x_label
            ](cond_feats)
        if self.embedding_head:
            for key in self.embedding_head.keys():
                z_parts[key] = self.embedding_head[key](z_parts[self.hparams.x_label])

        return z_parts

    def decode(self, z_parts, return_canonical=False, batch=None):
        if hasattr(self.encoder[self.hparams.x_label], "generate_grid_feats"):
            if self.encoder[self.hparams.x_label].generate_grid_feats:
                base_xhat = self.decoder[self.hparams.x_label](
                    batch[self.point_label], z_parts["grid_feats"]
                )
            else:
                base_xhat = self.decoder[self.hparams.x_label](
                    z_parts[self.hparams.x_label]
                )
        else:
            base_xhat = self.decoder[self.hparams.x_label](
                z_parts[self.hparams.x_label]
            )

        if self.get_rotation:
            rotation = z_parts["rotation"]
            xhat = torch.einsum("bij,bjk->bik", base_xhat[:, :, :3], rotation)
            if xhat.shape[-1] != base_xhat.shape[-1]:
                xhat = torch.cat([xhat, base_xhat[:, :, -1:]], dim=-1)
        else:
            xhat = base_xhat

        if return_canonical:
            return {self.hparams.x_label: xhat, "canonical": base_xhat}

        return {self.hparams.x_label: xhat}

    def decoder_compose_function(self, z_parts, batch):
        if self.condition_keys:
            for j, key in enumerate(self.condition_keys):
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

    def calculate_rcl_dict(self, x, xhat, z):
        rcl_per_input_dimension = {}
        rcl_reduced = {}
        for key in xhat.keys():
            rcl_per_input_dimension[key] = self.calculate_rcl(
                x, xhat, key, self.occupancy_label
            )
            if len(rcl_per_input_dimension[key].shape) > 0:
                rcl = (
                    rcl_per_input_dimension[key]
                    # flatten
                    .view(rcl_per_input_dimension[key].shape[0], -1)
                    # and sum across each batch element's dimensions
                    .sum(dim=1)
                )

                rcl_reduced[key] = rcl.mean()
            else:
                rcl_reduced[key] = rcl_per_input_dimension[key]

        if self.embedding_head_loss:
            for key in self.embedding_head_loss.keys():
                rcl_reduced[key] = self.embedding_head_weight[
                    key
                ] * self.embedding_head_loss[key](z[key], x[key])

        if self.basal_head_loss:
            for key in self.basal_head_loss.keys():
                rcl_reduced[key] = self.basal_head_weight[key] * self.basal_head_loss[
                    key
                ](z[key], x[key])
        return rcl_reduced

    def forward(self, batch, decode=False, inference=True, return_params=False):
        z_params = self.encode(batch, get_rotation=self.get_rotation)
        z_params = self.encoder_compose_function(z_params)
        z = self.sample_z(z_params, inference=inference)

        z = self.decoder_compose_function(z, batch)

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
