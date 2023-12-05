from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss as Loss

from cyto_dl.models.vae.base_vae import BaseVAE
from cyto_dl.models.vae.priors import IdentityPrior, IsotropicGaussianPrior
from cyto_dl.nn import MLP
from cyto_dl.utils.spharm import flip_spharm, get_indices, rotate_spharm

from .o2_spharm_encoder import O2SpharmEncoder

Array = Union[torch.Tensor, np.array, Sequence[float]]


class O2SpharmVAE(BaseVAE):
    def __init__(
        self,
        x_dim: int,
        latent_dim: int,
        encoder_hidden_layers: Sequence[int],
        decoder_hidden_layers: Sequence[int],
        x_label: str,
        reflections: bool = False,
        id_label: Optional[str] = None,
        optimizer=torch.optim.Adam,
        lr_scheduler=None,
        beta: float = 1.0,
        columns: Sequence[str] = None,
        max_spharm_band: int = 16,
        max_hidden_band: int = 8,
        grid_size: int = 64,
        reconstruction_loss: Loss = nn.MSELoss(reduction="none"),
        prior: str = "gaussian",
        eps: float = 1e-9,
    ):
        assert prior in ("gaussian", "none")

        self.eps = eps

        self.max_spharm_band = max_spharm_band
        self.flat_indices, self.paired_indices = get_indices(columns, max_spharm_band)
        self.reflections = reflections

        prior = {
            "embedding": (
                IsotropicGaussianPrior()
                if prior == "gaussian"
                else IdentityPrior(dimensionality=latent_dim)
            ),
            "angle": IdentityPrior(dimensionality=1),
        }

        if reflections:
            prior["flip"] = IdentityPrior(dimensionality=1)

        encoder = O2SpharmEncoder(
            hidden_layers=encoder_hidden_layers,
            reflections=reflections,
            out_dim=(2 * latent_dim if prior == "gaussian" else latent_dim),
            max_spharm_band=max_spharm_band,
            max_hidden_band=max_hidden_band,
            grid_size=grid_size,
        )
        decoder = MLP(
            latent_dim,
            x_dim,
            hidden_layers=decoder_hidden_layers,
        )

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            x_label=x_label,
            id_label=id_label,
            beta=beta,
            prior=prior,
            columns=columns,
            reconstruction_loss=reconstruction_loss,
        )

    def encode(self, batch):
        # reorder x's columns to match the encoder
        x = batch[self.hparams.x_label][:, self.flat_indices]

        # convert x to geometric tensor
        x = self.encoder[self.hparams.x_label].in_type(x)

        z = self.encoder[self.hparams.x_label](x).tensor

        parts = {
            "embedding": z[:, : 2 * self.latent_dim],
            "angle": z[:, 2 * self.latent_dim : 2 * self.latent_dim + 2],
        }

        parts["angle"] = parts["angle"] / (
            torch.norm(parts["angle"], dim=-1, keepdim=True) + self.eps
        )

        if self.reflections:
            z3 = z[:, 2 * self.latent_dim + 2 :]
            ortho_z2 = parts["angle"][:, [1, 0]] * torch.tensor((-1, 1), device=z.device)
            z3 = ortho_z2 * (z3 * ortho_z2).sum(axis=1).unsqueeze(1)
            z3 = z3 / (torch.norm(z3, dim=-1, keepdim=True) + self.eps)
            parts["flip"] = (
                parts["angle"][:, 0] * z3[:, 1] - parts["angle"][:, 1] * z3[:, 0]
            ).unsqueeze(1)

        return parts

    def decode(self, z_parts):
        base_xhat = self.decoder[self.hparams.x_label](z_parts["embedding"])
        angles = z_parts["angle"]

        if self.reflections:
            # flip the image to the canonical flip, and correct the rotation vector
            xhat = flip_spharm(base_xhat, self.paired_indices, flips=z_parts["flip"])
            _angles = torch.stack((angles[:, 0] * z_parts["flip"].squeeze(), angles[:, 1]), dim=1)
        else:
            _angles = angles

        xhat = rotate_spharm(
            base_xhat,
            _angles,
            self.paired_indices,
            self.max_spharm_band,
        )

        return {self.hparams.x_label: xhat}, z_parts
