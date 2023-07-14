import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from monai.networks.blocks import SubpixelUpsample
from monai.networks.layers.convutils import calculate_out_shape, same_padding
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import Flatten, Reshape
from monai.networks.nets import AutoEncoder
from omegaconf import DictConfig
from torch.nn.modules.loss import _Loss as Loss

from aics_im2im.image.transforms import O2Mask
from aics_im2im.models.vae.base_vae import BaseVAE
from aics_im2im.models.vae.priors import IdentityPrior, IsotropicGaussianPrior

from .modules_2d import Decoder as Decoder2D
from .modules_2d import Encoder as Encoder2D
from .modules_3d import Decoder as Decoder3D
from .modules_3d import Encoder as Encoder3D
from .so2_encoder import SO2ImageEncoder
from .utils import get_rotation_matrix, rotate_img

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


class SO2ImageVAE(BaseVAE):
    def __init__(
        self,
        latent_dim: int,
        spatial_dims: int,
        in_shape: Sequence[int],
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        x_label: str,
        id_label: Optional[str] = None,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        maximum_frequency: int = 8,
        hidden_dim: int = 32,
        kernel_size: int = 3,
        up_kernel_size: int = 3,
        num_res_units: int = 0,
        act: Optional[Union[Sequence[str], str]] = Act.PRELU,
        norm: Union[Sequence[str], str] = Norm.INSTANCE,
        dropout: Optional[Union[Sequence, str, float]] = None,
        bias: bool = True,
        beta: float = 1.0,
        reconstruction_loss: Loss = nn.MSELoss(reduction="none"),
        embedding_prior: str = "gaussian",
        use_sigmoid: bool = True,
        encoder_out_size: Optional[int] = None,
        encoder_relevance: bool = True,
        encoder_paddingmode: str = "replicate",
        decoder_pixelshuffle: bool = True,
        mask_input: bool = True,
        mask_output: bool = True,
        eps: float = 1e-5,
    ):
        self.spatial_dims = spatial_dims
        self.in_channels, *self.in_shape = in_shape
        self.use_sigmoid = use_sigmoid
        self.eps = eps
        self.hidden_dim = hidden_dim

        self.latent_dim = latent_dim
        self.final_size = np.asarray(self.in_shape, dtype=int)

        self.mask_input = mask_input
        self.mask_output = mask_output

        if mask_input or mask_output:
            self.mask = O2Mask(
                spatial_dims,
                max(self.in_shape[-2:]),
            )
        else:
            self.mask = None

        _full_net = AutoEncoder(
            spatial_dims,
            self.in_channels,
            out_channels,
            channels,
            strides,
            kernel_size,
            up_kernel_size,
            num_res_units,
            None,
            None,
            0,
            act,
            norm,
            dropout,
            bias,
        )

        del _full_net.encode  # we don't need the MONAI encoder

        self.kernel_size = kernel_size

        padding = same_padding(kernel_size)

        for s in strides:
            self.final_size = calculate_out_shape(self.final_size, kernel_size, s, padding)

        linear_size = int(np.product(self.final_size)) * channels[-1]

        decodeL = nn.Linear(self.latent_dim, linear_size)

        # encoder_out_size = encoder_out_size or latent_dim

        # encoder = SO2ImageEncoder(
        #     spatial_dims=spatial_dims,
        #     out_dim=encoder_out_size,
        #     channels=channels,
        #     strides=strides,
        #     maximum_frequency=maximum_frequency,
        #     kernel_size=kernel_size,
        #     bias=bias,
        #     relevance=encoder_relevance,
        #     padding_mode=encoder_paddingmode,
        # )
        if spatial_dims == 3:
            encoder = Encoder3D(
                encoder_out_size, hidden_dim=self.hidden_dim, pool=False, in_channel=1
            )
            decoder = Decoder3D(latent_dim, self.hidden_dim, in_channel=1)
        elif spatial_dims == 2:
            encoder = Encoder2D(
                encoder_out_size, hidden_dim=self.hidden_dim, pool=False, in_channel=1
            )
            decoder = Decoder2D(latent_dim, self.hidden_dim)
        else:
            raise Exception("Spatial dims must be 2 or 3")

        # if decoder_pixelshuffle:
        #     last_layer = SubpixelUpsample(
        #         self.spatial_dims,
        #         channels[0],
        #         out_channels=self.in_channels,
        #         scale_factor=strides[0],
        #         apply_pad_pool=True,
        #     )

        #     _full_net.decode[-1] = last_layer

        # decoder = nn.Sequential(
        #     decodeL,
        #     Reshape(_full_net.encoded_channels, *self.final_size),
        #     _full_net.decode,
        #     nn.Sigmoid() if use_sigmoid else nn.Identity(),
        # )

        encoder = {x_label: encoder}
        decoder = {x_label: decoder}

        if not isinstance(reconstruction_loss, (dict, DictConfig)):
            assert x_label is not None
            reconstruction_loss = {x_label: reconstruction_loss}

        prior = {
            "embedding": (
                IsotropicGaussianPrior()
                if embedding_prior == "gaussian"
                else IdentityPrior(dimensionality=latent_dim)
            ),
            "angle": IdentityPrior(dimensionality=1),
        }

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
        if self.mask_input:
            x = self.mask(x)

        z, pose = self.encoder[self.hparams.x_label](x)

        parts = {
            "embedding": z,
            "angle": pose,
        }

        parts["angle"] = parts["angle"] / (
            torch.norm(parts["angle"], dim=-1, keepdim=True) + self.eps
        )

        return parts

    def decode(self, z_parts, return_canonical=False):
        base_xhat = self.decoder[self.hparams.x_label](z_parts["embedding"])
        angles = z_parts["angle"]
        R = get_rotation_matrix(angles, spatial_dims=self.spatial_dims)
        xhat = rotate_img(base_xhat, R)

        if self.mask_output:
            xhat = self.mask(xhat)
        if return_canonical:
            return {self.hparams.x_label: xhat}, z_parts, base_xhat
        else:
            return {self.hparams.x_label: xhat}, z_parts
