import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from cyto_dl.models.vae.base_vae import BaseVAE
from cyto_dl.models.vae.priors import IdentityPrior, IsotropicGaussianPrior
from cyto_dl.nn.losses import ChamferLoss, L1Loss
from cyto_dl.nn.point_cloud import DGCNN, FoldingNet, LocalDecoder

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


class ModelVAE(BaseVAE):
    def __init__(
        self,
        model,
        x_label: str,
        recon_loss: Optional[dict] = None,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        **base_kwargs,
    ):
        super().__init__(
            encoder=None,
            decoder=None,
            latent_dim=None,
            x_label=x_label,
            optimizer=optimizer,
        )

        self.x_label = x_label
        if not isinstance(model, (dict, DictConfig)):
            assert x_label is not None
            self.model = {x_label: model}
        else:
            self.model = model
        self.model = nn.ModuleDict(self.model)

        if not isinstance(recon_loss, (dict, DictConfig)):
            assert x_label is not None
            self.rec_loss = {x_label: recon_loss}
        else:
            self.rec_loss = recon_loss
        self.rec_loss = nn.ModuleDict(self.rec_loss)

    def model_step(self, stage, batch, batch_idx):
        loss = self.forward(batch, decode=False)
    
        loss = {
            "loss": loss,
            "total_kld": loss.detach(),
            "total_reconstruction": loss.detach()}

        return loss, {}, None

    def forward(self, batch, decode=False):
        self.model[self.x_label] = self.model[self.x_label].to(
            batch[self.x_label].device
        )
        rec_points, gt_points = self.model[self.x_label](batch[self.x_label])
        loss = self.rec_loss[self.x_label](rec_points, gt_points)
        if decode:
            return loss, rec_points, gt_points
        else:
            return loss
