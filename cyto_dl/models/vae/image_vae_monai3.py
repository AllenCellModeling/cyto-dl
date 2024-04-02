import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from monai.networks.blocks import Convolution, ResidualUnit, UpSample
from monai.networks.layers.convutils import calculate_out_shape, same_padding
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import Flatten, Reshape
from omegaconf import DictConfig
from torch.nn.modules.loss import _Loss as Loss
from monai.networks.nets import AutoEncoder

from cyto_dl.image.transforms import RotationMask
from cyto_dl.models.vae.base_vae import BaseVAE
from cyto_dl.utils.rotation import RotationModule
from cyto_dl.models.vae.implicit_decoder import ImplicitDecoder, MultiplyConstant
from .utils import weight_init

from .image_encoder import ImageEncoder

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


class _Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = torch.tensor(scale)

    def forward(self, x):
        return x * self.scale.type_as(x)


class ImageVAEMonai3(BaseVAE):
    def __init__(
        self,
        x_label: str,
        latent_dim: int,
        spatial_dims: int,
        in_shape: Sequence[int],
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_sizes: Sequence[int],
        group: Optional[str] = None,
        out_channels: int = None,
        decoder_initial_shape: Optional[Sequence[int]] = None,
        decoder_channels: Optional[Sequence[int]] = None,
        decoder_strides: Optional[Sequence[int]] = None,
        maximum_frequency: int = 8,
        background_value: float = 0,
        act: Optional[Union[Sequence[str], str]] = Act.PRELU,
        norm: Union[Sequence[str], str] = Norm.INSTANCE,
        dropout: Optional[Union[Sequence, str, float]] = None,
        bias: bool = True,
        prior: str = "gaussian",
        last_act: Optional[str] = None,
        last_scale: float = 1.0,
        mask_input: bool = False,
        mask_output: bool = False,
        clip_min: Optional[int] = None,
        clip_max: Optional[int] = None,
        num_res_units: int = 2,
        up_kernel_size: int = 3,
        first_conv_padding_mode: str = "replicate",
        eps: float = 1e-8,
        encoder_padding: Optional[Union[int, Sequence[int]]] = None,
        metric_keys: Optional[list] = None,
        use_implicit_decoder: Optional[bool] = False,
        decoder_res_units: Optional[int] = None,
        override_final_size: Optional[tuple] = None,
        **base_kwargs,
    ):
        in_channels, *in_shape = in_shape
        if decoder_res_units is None:
            decoder_res_units = num_res_units
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.x_label = x_label
        self.spatial_dims = spatial_dims
        self.final_size = np.asarray(in_shape, dtype=int)
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.use_implicit_decoder = use_implicit_decoder
        self.act = act
        self.norm = norm
        self.bias = bias
        self.dropout = dropout

        self.mask_input = mask_input
        self.mask_output = mask_output

        monai_autoencoder_ = AutoEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            channels=channels,
            strides=strides,
            kernel_size=3,
            up_kernel_size=3,
            num_res_units=num_res_units,
            act="relu",
            norm="batch",
            # dropout: tuple | str | float | None = None,
            # bias: bool = True,
            # padding: Sequence[int] | int | None = None,
        )
        encoder = monai_autoencoder_.encode
        decoder = monai_autoencoder_.decode

        if group is not None:
            self.rotation_module = RotationModule(
                group, spatial_dims, background_value, eps
            )
        else:
            self.rotation_module = None

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            prior=prior,
            x_label=x_label,
            metric_keys=metric_keys,
            **base_kwargs,
        )

    def encode(self, batch):
        x = batch[self.hparams.x_label]
        if self.mask_input:
            x = self.mask(x)

        if self.hparams.group is not None:
            z, pose = self.encoder["embedding"](x)
            parts = {"embedding": z, "pose": pose}
        else:
            z = self.encoder["embedding"](x)
            parts = {"embedding": z}

        if self.hparams.group == "so2":
            parts["pose"] = parts["pose"] / (
                torch.norm(parts["pose"], dim=-1, keepdim=True) + self.hparams.eps
            )

        return parts

    def decode(self, z_parts, return_canonical=False):
        pool_dims = (2, 3) if self.spatial_dims == 2 else (2, 3, 4)
        z_parts["embedding"] = z_parts["embedding"].mean(dim=pool_dims)
        z_parts["embedding"] = (
            z_parts["embedding"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )  # [bz, emb_dim, 1, 1, 1]
        z_parts["embedding"] = z_parts["embedding"].expand(-1, -1, 4, 4, 4)

        base_xhat = self.decoder[self.hparams.x_label](z_parts["embedding"])
        clip_min = self.hparams.get("clip_min")
        clip_max = self.hparams.get("clip_min")

        if clip_min is not None or clip_max is not None:
            base_xhat = base_xhat.clip(clip_min, clip_max)

        if self.hparams.group is not None:
            xhat = self.rotation_module(base_xhat, z_parts["pose"])
        else:
            xhat = base_xhat

        if self.mask_output:
            xhat = self.mask(xhat)

        if return_canonical:
            return {self.hparams.x_label: xhat, "canonical": base_xhat}

        return {self.hparams.x_label: xhat}
