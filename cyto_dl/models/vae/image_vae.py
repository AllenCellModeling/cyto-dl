import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, ResidualUnit, UpSample
from monai.networks.layers.convutils import calculate_out_shape, same_padding
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import Reshape

from cyto_dl.image.transforms import RotationMask
from cyto_dl.models.vae.base_vae import BaseVAE
from cyto_dl.utils.rotation import RotationModule
from cyto_dl.models.vae.implicit_decoder import ImplicitDecoder
from .utils import weight_init

from .image_encoder import ImageEncoder
from monai.networks.nets import AutoEncoder

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


class _Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = torch.tensor(scale)

    def forward(self, x):
        return x * self.scale.type_as(x)


class _Interpolate(nn.Module):
    def __init__(self, size, mode="trilinear", align_corners=False):
        super(_Interpolate, self).__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            x, size=self.size, mode=self.mode, align_corners=self.align_corners
        )


class ImageVAE(BaseVAE):
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
        average_spatial: Optional[bool] = True,
        use_monai_decoder: Optional[bool] = False,
        decoder_use_upsample: Optional[bool] = True,
        decoder_padding: Optional[list] = None,
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
        if decoder_use_upsample:
            for i, (s, c_in, c_out) in enumerate(
                zip(_strides, _channels[:-1], _channels[1:])
            ):
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
                    subunits=decoder_res_units,
                    padding=1,
                )
                decode_blocks.append(nn.Sequential(upsample, res))
        else:
            for i, (s, c_in, c_out) in enumerate(
                zip(_strides, _channels[:-1], _channels[1:])
            ):
                last_block = i == len(_strides) - 1

                conv = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=c_in,
                    out_channels=c_out,
                    strides=s,
                    kernel_size=3,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                    bias=bias,
                    padding=None if decoder_padding is None else decoder_padding[i],
                    conv_only=last_block and decoder_res_units == 0,
                    is_transposed=True,
                )

                if decoder_res_units > 0:
                    res = ResidualUnit(
                        spatial_dims=spatial_dims,
                        in_channels=c_out,
                        out_channels=c_out,
                        strides=1,
                        kernel_size=3,
                        act=act,
                        norm=norm,
                        dropout=dropout,
                        subunits=1,
                        padding=0,
                        last_conv_only=last_block,
                    )
                    if last_block:
                        decode_blocks.append(
                            nn.Sequential(
                                conv,
                                res,
                                _Interpolate(
                                    size=in_shape, mode="trilinear", align_corners=False
                                ),
                            )
                        )
                    else:
                        decode_blocks.append(nn.Sequential(conv, res))
                else:
                    decode_blocks.append(conv)

        init_shape = (
            self.final_size if decoder_initial_shape is None else decoder_initial_shape
        )

        if average_spatial:
            first_upsample = nn.Sequential(
                nn.Linear(latent_dim, _channels[0] * int(np.product(init_shape))),
                Reshape(_channels[0], *init_shape),
            )
        else:
            first_upsample = nn.Sequential(torch.nn.Identity())

        if use_implicit_decoder:
            # decoder_hidden_channels = [64, 64, 64, 64, 1]
            # final_non_linearity = nn.Sequential(
            #     torch.nn.Tanh(),
            #     MultiplyConstant(constant=2),
            # )
            decoder_hidden_channels = [32, 32, 32, 32, 1]
            final_non_linearity = nn.Sequential()
            decoder = ImplicitDecoder(
                latent_dims=latent_dim,
                hidden_channels=decoder_hidden_channels,
                non_linearity=torch.nn.ReLU(),
                final_non_linearity=final_non_linearity,
                mode="3d",
            )
            decoder.apply(weight_init)
        elif not decoder_use_upsample:
            decoder = nn.Sequential(
                first_upsample,
                *decode_blocks,
            )

        else:
            decoder = nn.Sequential(
                first_upsample,
                *decode_blocks,
                # decoder,
                last_act if last_act is not None else nn.Identity(),
                _Scale(last_scale),
            )

        if use_monai_decoder:
            monai_autoencoder_ = AutoEncoder(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=in_channels,
                channels=channels,
                strides=strides,
                kernel_size=3,
                up_kernel_size=3,
                num_res_units=num_res_units,
                act=self.act,
                norm="batch",
                # dropout: tuple | str | float | None = None,
                # bias: bool = True,
                # padding: Sequence[int] | int | None = None,
            )
            decoder = monai_autoencoder_.decode

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
            average_spatial=average_spatial,
        )

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
