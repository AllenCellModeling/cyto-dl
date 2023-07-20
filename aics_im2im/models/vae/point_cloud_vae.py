import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from aics_im2im.models.vae.base_vae import BaseVAE
from aics_im2im.models.vae.priors import IdentityPrior, IsotropicGaussianPrior
from aics_im2im.nn.losses import ChamferLoss, L1Loss
from aics_im2im.nn.point_cloud import DGCNN, FoldingNet, LocalDecoder

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
        hidden_conv2d_channels: list = [64, 64, 64, 64],
        hidden_conv1d_channels: list = [512, 20],
        hidden_decoder_dim: int = 512,
        k=20,
        mode="scalar",
        get_rotation=False,
        include_cross=True,
        include_coords=True,
        id_label: Optional[str] = None,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        beta: float = 1.0,
        embedding_prior: str = "identity",
        decoder_type: str = "foldingnet",
        loss_type: str = "chamfer",
        eps: float = 1e-6,
        shape: str = "sphere",
        num_coords: int = 3,
        std: float = 0.3,
        sphere_path: str = "/allen/aics/modeling/ritvik/projects/cellshape/cellshape-cloud/cellshape_cloud/vendor/sphere.npy",
        gaussian_path: str = "/allen/aics/modeling/ritvik/projects/cellshape/cellshape-cloud/cellshape_cloud/vendor/gaussian.npy",
        symmetry_breaking_axis: Optional[Union[str, int]] = None,
        scalar_inds: Optional[int] = None,
        generate_grid_feats: Optional[bool] = False,
        padding: Optional[float] = 0.1,
        reso_plane: Optional[int] = 64,
        plane_type: Optional[list] = ["xz", "xy", "yz"],
        scatter_type: Optional[str] = "max",
        point_label: Optional[str] = "points",
        occupancy_label: Optional[str] = "points.df",
        **base_kwargs,
    ):
        self.get_rotation = get_rotation or mode == "vector"
        self.symmetry_breaking_axis = symmetry_breaking_axis
        self.scalar_inds = scalar_inds
        self.decoder_type = decoder_type
        self.generate_grid_feats = generate_grid_feats
        self.occupancy_label = occupancy_label
        self.point_label = point_label

        if embedding_prior == "gaussian":
            self.encoder_out_size = 2 * latent_dim
        else:
            self.encoder_out_size = latent_dim

        encoder = DGCNN(
            num_features=self.encoder_out_size,
            hidden_dim=hidden_dim,
            hidden_conv2d_channels=hidden_conv2d_channels,
            hidden_conv1d_channels=hidden_conv1d_channels,
            k=k,
            mode=mode,
            scalar_inds=scalar_inds,
            include_cross=include_cross,
            include_coords=include_coords,
            symmetry_breaking_axis=symmetry_breaking_axis,
            generate_grid_feats=generate_grid_feats,
            padding=padding,
            reso_plane=reso_plane,
            plane_type=plane_type,
            scatter_type=scatter_type,
        )
        if decoder_type == "foldingnet":
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
        elif decoder_type == "localdecoder":
            decoder = LocalDecoder(latent_dim, hidden_decoder_dim)

        encoder = {x_label: encoder}
        decoder = {x_label: decoder}
        if loss_type == "chamfer":
            reconstruction_loss = {x_label: ChamferLoss()}
        elif loss_type == "L1":
            reconstruction_loss = {x_label: L1Loss()}

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

        if self.generate_grid_feats:
            embedding, rotation, grid_feats = self.encoder[self.hparams.x_label](
                x, get_rotation=self.get_rotation
            )
            return {
                "embedding": embedding,
                "rotation": rotation,
                "grid_feats": grid_feats,
            }

        return {"embedding": self.encoder[self.hparams.x_label](x)}

    def decode(self, z_parts, return_canonical=False, batch=None):
        if self.generate_grid_feats:
            base_xhat = self.decoder[self.hparams.x_label](
                batch[self.point_label], z_parts["grid_feats"]
            )
        else:
            base_xhat = self.decoder[self.hparams.x_label](z_parts["embedding"])

        if self.get_rotation:
            rotation = z_parts["rotation"]
            xhat = torch.einsum("bij,bjk->bik", base_xhat, rotation)
        else:
            xhat = base_xhat

        if return_canonical:
            return {self.hparams.x_label: xhat}, base_xhat

        return {self.hparams.x_label: xhat}

    def calculate_elbo(self, x, xhat, z):
        rcl_reduced = self.calculate_rcl_dict(x, xhat, self.occupancy_label)

        kld_per_part = {
            part: prior(z[part], mode="kl", reduction="none")
            for part, prior in self.prior.items()
        }

        kld_per_part_summed = {
            part: kl.sum(dim=-1).mean() for part, kl in kld_per_part.items()
        }

        total_kld = sum(kld_per_part_summed.values())
        total_recon = sum(rcl_reduced.values())
        return (
            total_recon + self.beta * total_kld,
            total_recon,
            rcl_reduced,
            total_kld,
            kld_per_part,
        )

    def forward(self, batch, decode=False, inference=True, return_params=False):
        is_inference = inference or not self.training

        z_params = self.encode(batch)
        z = self.sample_z(z_params, inference=inference)

        if not decode:
            return z

        if self.generate_grid_feats:
            xhat = self.decode(z, batch=batch)
        else:
            xhat = self.decode(z)

        if return_params:
            return xhat, z, z_params

        return xhat, z
