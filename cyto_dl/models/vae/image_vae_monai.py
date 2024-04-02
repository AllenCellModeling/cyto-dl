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

from cyto_dl.image.transforms import RotationMask
from cyto_dl.models.vae.base_vae import BaseVAE
from cyto_dl.utils.rotation import RotationModule
from cyto_dl.models.vae.implicit_decoder import ImplicitDecoder, MultiplyConstant
from .utils import weight_init
from monai.networks.nets import AutoEncoder

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


class ImageVAEMonai(BaseVAE):
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
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.x_label = x_label
        self.spatial_dims = spatial_dims
        self.final_size = np.asarray(in_shape, dtype=int)
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.bias = bias
        self.dropout = dropout

        self.mask_input = mask_input
        self.mask_output = mask_output

        if last_act is not None:
            if last_act == "sigmoid":
                last_act = nn.Sigmoid()
            elif last_act == "tanh":
                last_act = nn.Tanh()
            else:
                raise ValueError("`last_act` must be either 'sigmoid' or 'tanh'")

        if mask_input or mask_output:
            if group is not None:
                self.mask = RotationMask(
                    group,
                    spatial_dims,
                    max(in_shape[-2:]),
                    background=background_value,
                )
            else:
                self.mask = None
                self.mask_input = None
                self.mask_output = None
        else:
            self.mask = None

        if encoder_padding is None:
            encoder_padding = [None] * len(kernel_sizes)

        for k, s, p in zip(kernel_sizes, strides, encoder_padding):
            padding = same_padding(k) if p is None else p
            self.final_size = calculate_out_shape(self.final_size, k, s, padding)

            if override_final_size:
                self.final_size = override_final_size

        if isinstance(prior, (str, type(None))):
            if prior == "gaussian":
                encoder_out_size = 2 * latent_dim
            else:
                encoder_out_size = latent_dim
        else:
            encoder_out_size = prior.param_size

        encoder = ImageEncoder(
            spatial_dims=spatial_dims,
            out_dim=encoder_out_size,
            channels=channels,
            strides=strides,
            maximum_frequency=maximum_frequency,
            kernel_sizes=kernel_sizes,
            bias=bias,
            padding=encoder_padding,
            group=group,
            first_conv_padding_mode=first_conv_padding_mode,
            num_res_units=num_res_units,
        )

        if group is not None:
            self.rotation_module = RotationModule(
                group, spatial_dims, background_value, eps
            )
        else:
            self.rotation_module = None

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
        decoder = monai_autoencoder_.decode

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
