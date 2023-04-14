from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss

from aics_im2im.nn import MLP

from .base_vae import BaseVAE
from .priors import Prior

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]


class TabularVAE(BaseVAE):
    def __init__(
        self,
        x_dim: int,
        latent_dim: int,
        hidden_layers: Sequence[int],
        x_label: str,
        id_label: Optional[str] = None,
        loss_mask_label: Optional[str] = None,
        optimizer=torch.optim.Adam,
        beta: float = 1.0,
        priors: Optional[Sequence[Prior]] = None,
        reconstruction_loss: Loss = nn.MSELoss(reduction="none"),
        reconstruction_reduce: str = "sum",
    ):
        encoder = MLP(
            x_dim,
            2 * latent_dim,
            hidden_layers=hidden_layers,
        )

        decoder = MLP(
            latent_dim,
            x_dim,
            hidden_layers=hidden_layers,
        )

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            optimizer=optimizer,
            x_label=x_label,
            id_label=id_label,
            loss_mask_label=loss_mask_label,
            beta=beta,
            priors=priors,
            reconstruction_loss=reconstruction_loss,
            reconstruction_reduce=reconstruction_reduce,
        )
