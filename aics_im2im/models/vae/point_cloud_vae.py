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
        hidden_dim=64,
        hidden_decoder_dim=512,
        k=20,
        mode="scalar",
        get_rotation=False,
        include_cross=True,
        include_coords=True,
        id_label: Optional[str] = None,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        beta: float = 1.0,
        embedding_prior: str = "identity",
        eps: float = 1e-6,
        shape: str = "sphere",
        num_coords: int = 3,
        std: float = 0.3,
        sphere_path: str = "/allen/aics/modeling/ritvik/projects/cellshape/cellshape-cloud/cellshape_cloud/vendor/sphere.npy",
        gaussian_path: str = "/allen/aics/modeling/ritvik/projects/cellshape/cellshape-cloud/cellshape_cloud/vendor/gaussian.npy",
        symmetry_breaking_axis: Optional[Union[str, int]] = None,
        scalar_inds: Optional[int] = None,
        **base_kwargs,
    ):
        self.get_rotation = get_rotation or mode == "vector"
        self.symmetry_breaking_axis = symmetry_breaking_axis
        self.scalar_inds = scalar_inds

        if embedding_prior == "gaussian":
            self.encoder_out_size = 2 * latent_dim
        else:
            self.encoder_out_size = latent_dim

        encoder = DGCNN(
            num_features=self.encoder_out_size,
            hidden_dim=hidden_dim,
            k=k,
            mode=mode,
            scalar_inds=scalar_inds,
            include_cross=include_cross,
            include_coords=include_coords,
            get_rotation=get_rotation,
            symmetry_breaking_axis=symmetry_breaking_axis,
        )

        decoder = FoldingNet(
            latent_dim,
            num_points,
            hidden_decoder_dim,
            std,
            shape,
            sphere_path,
            gaussian_path,
            num_coords,
        )

        encoder = {x_label: encoder}
        decoder = {x_label: decoder}
        reconstruction_loss = {x_label: ChamferLoss()}

        prior = {
            "embedding": (
                IsotropicGaussianPrior(dimensionality=latent_dim)
                if embedding_prior == "gaussian"
                else IdentityPrior(dimensionality=latent_dim)
            ),
        }
        if self.get_rotation:
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

        if self.get_rotation:
            embedding, rotation = self.encoder[self.hparams.x_label](
                x, get_rotation=self.get_rotation
            )
            return {"embedding": embedding, "rotation": rotation}

        return {"embedding": self.encoder[self.hparams.x_label](x)}

    def decode(self, z_parts, return_canonical=False):
        base_xhat = self.decoder[self.hparams.x_label](z_parts["embedding"])

        if self.get_rotation:
            rotation = z_parts["rotation"]
            xhat = torch.einsum("bij,bjk->bik", base_xhat, rotation)
        else:
            xhat = base_xhat

        if return_canonical:
            return {self.hparams.x_label: xhat}, base_xhat
        else:
            return {self.hparams.x_label: xhat}
