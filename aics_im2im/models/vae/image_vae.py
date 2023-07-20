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

from .base_vae import BaseVAE
from .priors import IsotropicGaussianPrior, Prior

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


class ImageVAE(BaseVAE):
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
        kernel_size: int = 3,
        up_kernel_size: int = 3,
        num_res_units: int = 0,
        act: Optional[Union[Sequence[str], str]] = Act.PRELU,
        norm: Union[Sequence[str], str] = Norm.INSTANCE,
        dropout: Optional[Union[Sequence, str, float]] = None,
        bias: bool = True,
        beta: float = 1.0,
        reconstruction_loss: Loss = nn.MSELoss(reduction="none"),
        prior: Optional[Prior] = None,
        use_sigmoid: bool = True,
        encoder_out_size: Optional[int] = None,
        decoder_pixelshuffle: bool = True,
        **base_kwargs,
    ):
        self.in_channels, *self.in_shape = in_shape
        self.use_sigmoid = use_sigmoid

        self.latent_dim = latent_dim
        self.final_size = np.asarray(self.in_shape, dtype=int)

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

        self.kernel_size = _full_net.kernel_size

        padding = same_padding(kernel_size)

        for s in strides:
            self.final_size = calculate_out_shape(
                self.final_size, self.kernel_size, s, padding
            )

        linear_size = int(np.product(self.final_size)) * _full_net.encoded_channels

        encoder_out_size = encoder_out_size or latent_dim
        encodeL = nn.Linear(linear_size, encoder_out_size)
        decodeL = nn.Linear(self.latent_dim, linear_size)

        encoder = nn.Sequential(
            _full_net.encode,
            Flatten(),
            encodeL,
        )

        if decoder_pixelshuffle:
            last_layer = SubpixelUpsample(
                spatial_dims,
                channels[0],
                out_channels=self.in_channels,
                scale_factor=strides[0],
                apply_pad_pool=True,
            )

            _full_net.decode[-1] = last_layer

        decoder = nn.Sequential(
            decodeL,
            Reshape(_full_net.encoded_channels, *self.final_size),
            _full_net.decode,
            nn.Sigmoid() if use_sigmoid else nn.Identity(),
        )

        encoder = {x_label: encoder}
        decoder = {x_label: decoder}

        if not isinstance(reconstruction_loss, (dict, DictConfig)):
            assert x_label is not None
            reconstruction_loss = {x_label: reconstruction_loss}

        if not isinstance(prior, (dict, DictConfig)):
            assert x_label is not None
            prior = {x_label: prior}

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
