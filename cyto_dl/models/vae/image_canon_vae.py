import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from monai.networks.blocks import ResidualUnit, UpSample
from monai.networks.layers.convutils import calculate_out_shape, same_padding
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import Flatten, Reshape
from omegaconf import DictConfig
from torch.nn.modules.loss import _Loss as Loss

from cyto_dl.image.transforms import RotationMask
from cyto_dl.models.vae.base_vae import BaseVAE
from cyto_dl.utils.rotation import RotationModule

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


class _Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = torch.tensor(scale)

    def forward(self, x):
        return x * self.scale.type_as(x)


class ImageCanonicalVAE(BaseVAE):
    def __init__(
        self,
        latent_dim: int,
        spatial_dims: int,
        in_shape: Sequence[int],
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_sizes: Sequence[int],
        group: Optional[str] = None,
        background_value: float = 0,
        out_channels: int = None,
        decoder_initial_shape: Optional[Sequence[int]] = None,
        decoder_channels: Optional[Sequence[int]] = None,
        decoder_strides: Optional[Sequence[int]] = None,
        act: Optional[Union[Sequence[str], str]] = Act.PRELU,
        norm: Union[Sequence[str], str] = Norm.INSTANCE,
        dropout: Optional[Union[Sequence, str, float]] = None,
        bias: bool = True,
        prior: str = "gaussian",
        last_act: Optional[str] = None,
        last_scale: float = 1.0,
        mask: bool = True,
        clip_min: Optional[int] = None,
        clip_max: Optional[int] = None,
        num_res_units: int = 2,
        up_kernel_size: int = 3,
        encoder_padding: Optional[Union[int, Sequence[int]]] = None,
        eps: float = 1e-8,
        **base_kwargs
    ):
        in_channels, *in_shape = in_shape

        self.out_channels = out_channels if out_channels is not None else in_channels

        self.spatial_dims = spatial_dims
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.bias = bias
        self.dropout = dropout

        self.mask = mask

        if last_act is not None:
            if last_act == "sigmoid":
                last_act = nn.Sigmoid()
            elif last_act == "tanh":
                last_act = nn.Tanh()
            else:
                raise ValueError("`last_act` must be either 'sigmoid' or 'tanh'")

        if mask:
            if group is not None:
                self.mask = RotationMask(
                    group,
                    spatial_dims,
                    max(in_shape[-2:]),
                    background=background_value,
                )
            else:
                self.mask = None
        else:
            self.mask = None

        final_size = np.asarray(in_shape, dtype=int)
        encode_blocks = []
        for k, s, p, c_in, c_out in zip(
            kernel_sizes, strides, encoder_padding, channels[:-1], channels[1:]
        ):
            padding = same_padding(k) if p is None else p
            final_size = calculate_out_shape(final_size, k, s, padding)

            encode_blocks.append(
                ResidualUnit(
                    spatial_dims=spatial_dims,
                    in_channels=c_in,
                    out_channels=c_out,
                    strides=s,
                    kernel_size=k,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                    subunits=num_res_units,
                    padding=padding,
                )
            )

        if isinstance(prior, (str, type(None))):
            if prior == "gaussian":
                encoder_out_size = 2 * latent_dim
            else:
                encoder_out_size = latent_dim
        else:
            encoder_out_size = prior.param_size

        encoder = nn.Sequential(
            *encode_blocks, Flatten(), nn.Linear(final_size * channels[-1], encoder_out_size)
        )

        if decoder_channels is None:
            _channels = channels[::-1]
        else:
            _channels = decoder_channels
        _channels += [self.out_channels]

        if decoder_strides is None:
            _strides = strides[::-1]
        else:
            _strides = decoder_strides

        assert len(_strides) + 1 == len(_channels)

        decode_blocks = []
        for i, (s, c_in, c_out) in enumerate(zip(_strides, _channels[:-1], _channels[1:])):
            last_block = i + 1 == len(_strides)

            size = None if not last_block else in_shape

            upsample = UpSample(
                spatial_dims=spatial_dims,
                in_channels=c_in,
                out_channels=c_in,
                scale_factor=s,  # ignored if size isn't None, i.e. in the last block
                size=size,
                kernel_size=3,
                pre_conv=None,
                # choices inspired by this article:
                # https://distill.pub/2016/deconv-checkerboard/
                mode="nontrainable",
                interp_mode="nearest",
                align_corners=None,
            )

            res = ResidualUnit(
                spatial_dims=spatial_dims,
                in_channels=c_in,
                out_channels=c_out,
                strides=1,
                kernel_size=3,
                act=act,
                norm=norm,
                dropout=dropout,
                subunits=num_res_units,
                padding=1,
            )

            decode_blocks.append(nn.Sequential(upsample, res))

        init_shape = final_size if decoder_initial_shape is None else decoder_initial_shape

        decoder = nn.Sequential(
            nn.Linear(latent_dim, _channels[0] * int(np.product(init_shape))),
            Reshape(_channels[0], *init_shape),
            *decode_blocks,
            last_act if last_act is not None else nn.Identity(),
            _Scale(last_scale)
        )

        if group is not None:
            self.rotation_module = RotationModule(group, spatial_dims, background_value, eps)
        else:
            self.rotation_module = None

        super().__init__(
            encoder=encoder, decoder=decoder, latent_dim=latent_dim, prior=prior, **base_kwargs
        )

    def canonicalize(self, x):
        pose = self.canonicalization(x).tensor

        if self.hparams.group == "so2":
            pose = pose / (torch.norm(pose, dim=-1, keepdim=True) + self.hparams.eps)

        x = self.rotation_module(x, pose)

        return x, pose

    def encode(self, batch):
        x = batch[self.hparams.x_label]
        if self.mask is not None:
            x = self.mask(x)

        x, pose = self.canonicalize(x)

        if self.hparams.group is not None:
            base_x = self.rotation_module(x, pose.transpose(1, 2))
            z = self.encoder["embedding"](base_x)
            parts = {"embedding": z, "pose": pose}
        else:
            z = self.encoder["embedding"](x)
            parts = {"embedding": z}

        return parts

    def decode(self, z, return_canonical=False):
        base_xhat = self.decoder[self.hparams.x_label](z["embedding"])
        clip_min = self.hparams.get("clip_min")
        clip_max = self.hparams.get("clip_min")

        if clip_min is not None or clip_max is not None:
            base_xhat = base_xhat.clip(clip_min, clip_max)

        if self.hparams.group is not None:
            xhat = self.rotation_module(base_xhat, z["pose"])
        else:
            xhat = base_xhat

        if self.mask is not None:
            xhat = self.mask(xhat)

        if return_canonical:
            return {self.hparams.x_label: xhat, "canonical": base_xhat}

        return {self.hparams.x_label: xhat}
