import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from aics_im2im.models.vae.base_vae import BaseVAE
from aics_im2im.models.vae.priors import IdentityPrior, IsotropicGaussianPrior
from aics_im2im.nn.losses import ChamferLoss
from aics_im2im.nn.point_cloud import DGCNN, FoldingNet

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


class PointCloudVAE(BaseVAE):
    def __init__(
        self,
        latent_dim: int,
        x_label: str,
        num_points: int,
        hidden_encoder_features=[64, 128, 256, 512],
        hidden_decoder_dim=512,
        k=20,
        mode="scalar",
        equivariant=False,
        include_cross=True,
        include_coords=True,
        id_label: Optional[str] = None,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        beta: float = 1.0,
        embedding_prior: str = "identity",
        eps: float = 1e-6,
        symmetry_breaking_axis: Optional[Union[str, int]] = None,
        _transpose_rotation: bool = False,
    ):
        self.equivariant = equivariant
        self.symmetry_breaking_axis = symmetry_breaking_axis
        self._transpose_rotation = _transpose_rotation

        if embedding_prior == "gaussian":
            self.encoder_out_size = 2 * latent_dim
        else:
            self.encoder_out_size = latent_dim

        encoder = DGCNN(
            num_features=self.encoder_out_size,
            hidden_features=hidden_encoder_features,
            k=k,
            mode=mode,
            include_cross=include_cross,
            include_coords=include_coords,
            get_rotation=equivariant,
            break_symmetry=(symmetry_breaking_axis is not None),
        )

        decoder = FoldingNet(num_points, hidden_decoder_dim)

        encoder = {x_label: encoder}
        decoder = {x_label: decoder}
        reconstruction_loss = {x_label: ChamferLoss()}

        prior = {
            "embedding": (
                IsotropicGaussianPrior()
                if embedding_prior == "gaussian"
                else IdentityPrior(dimensionality=latent_dim)
            ),
        }
        if self.equivariant:
            prior["rotation"] = IdentityPrior(dimensionality=1)

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
        )

    def encode(self, batch):
        x = batch[self.hparams.x_label]

        if self.equivariant:
            if isinstance(self.symmetry_breaking_axis, int):
                axis = self.symmetry_breaking_axis
            elif isinstance(self.symmetry_breaking_axis, str):
                axis = batch[self.symmetry_breaking_axis]
            else:
                axis = None

            embedding, rotation = self.encoder(x, symmetry_breaking_axis=axis)
            return {"embedding": embedding, "rotation": rotation}

        return {"embedding": self.encoder(x)}

    def decode(self, z_parts):
        base_xhat = self.decoder[self.hparams.x_label](z_parts["embedding"])

        if self.equivariant:
            rotation = z_parts["rotation"]
            if self._transpose_rotation:
                rotation = rotation.mT

            xhat = torch.einsum("bij,bjk->bik", base_xhat, rotation)
        else:
            xhat = base_xhat

        return {self.hparams.x_label: xhat}, z_parts
