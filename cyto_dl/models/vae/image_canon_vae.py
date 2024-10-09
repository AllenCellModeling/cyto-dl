import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as enn
from monai.networks.blocks import UpSample
from monai.networks.layers.convutils import calculate_out_shape, same_padding
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import Flatten, Reshape

from cyto_dl.image.transforms import RotationMask
from cyto_dl.models.vae.base_vae import BaseVAE
from cyto_dl.nn import ResidualUnit
from cyto_dl.utils.rotation import RotationModule

from .image_encoder import Convolution as EqConvolution

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
        maximum_frequency: int = 8,
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
        **base_kwargs,
    ):
        in_channels, *in_shape = in_shape

        self.out_channels = out_channels if out_channels is not None else in_channels

        self.group = group
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
        _channels = [in_channels] + channels
        for k, s, p, c_in, c_out in zip(
            kernel_sizes, strides, encoder_padding, _channels[:-1], _channels[1:]
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

        if prior is None or isinstance(prior, str):
            if prior == "gaussian":
                encoder_out_size = 2 * latent_dim
            else:
                encoder_out_size = latent_dim
        else:
            encoder_out_size = prior.param_size

        encoder = nn.Sequential(
            *encode_blocks,
            Flatten(),
            nn.Linear(np.prod(final_size) * channels[-1], encoder_out_size),
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
            _Scale(last_scale),
        )

        if group is not None:
            self.rotation_module = RotationModule(group, spatial_dims, background_value, eps)
        else:
            self.rotation_module = None

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            prior=prior,
            **base_kwargs,
        )

        self.make_canon_net(maximum_frequency)

    def make_canon_net(self, maximum_frequency):
        group = self.group
        spatial_dims = self.spatial_dims

        if group not in ("so2", "so3", None):
            raise ValueError(f"`group` should be one of ('so2', 'so3', None). Got {group!r}")

        if group == "so2":
            self.gspace = (
                gspaces.rot2dOnR2(N=-1, maximum_frequency=maximum_frequency)
                if self.spatial_dims == 2
                else gspaces.rot2dOnR3(n=-1, maximum_frequency=maximum_frequency)
            )
        elif group == "so3":
            if self.spatial_dims != 3:
                raise ValueError("The SO3 group only works for spatial_dims=3")
            gspace = gspaces.rot3dOnR3(maximum_frequency=maximum_frequency)
        else:
            gspace = gspaces.trivialOnR2() if spatial_dims == 2 else gspaces.trivialOnR3()

        in_type = enn.FieldType(gspace, [gspace.trivial_repr])

        if group == "so2":
            n_out_vectors = 1
        else:
            n_out_vectors = 2

        conv1 = EqConvolution(spatial_dims, in_type, 64, 1, 7, padding=0)
        conv_class = enn.R3Conv if spatial_dims == 3 else enn.R2Conv
        conv2 = conv_class(
            conv1.out_type,
            enn.FieldType(gspace, n_out_vectors * [gspace.irrep(1)]),
            kernel_size=7,
            stride=1,
        )
        self.canonicalization = enn.SequentialModule(conv1, conv2)

    def canonicalize(self, x):
        _x = self.canonicalization.in_type(x)
        pose = self.canonicalization(_x).tensor

        pool_dims = (2, 3) if self.spatial_dims == 2 else (2, 3, 4)
        pose = pose.mean(dim=pool_dims)

        if self.group == "so3":
            # separate two vectors into two channels
            pose = pose.reshape(pose.shape[0], 2, -1)

            # move from y z x (spharm) convention to x y z
            pose = pose[:, :, [2, 0, 1]]

        elif self.group == "so2":
            # move from y x (spharm) convention to x y
            pose = pose[:, [1, 0]]
            pose = pose / (torch.norm(pose, dim=-1, keepdim=True) + self.hparams.eps)

        pose = self.rotation_module.compute_rotation_matrix(pose)

        return x, pose

    def encode(self, batch):
        x = batch[self.hparams.x_label]
        if self.mask is not None:
            x = self.mask(x)

        x, pose = self.canonicalize(x)

        if self.hparams.group is not None:
            base_x = self.rotation_module(x, None, R=pose.transpose(1, 2))
            z = self.encoder["embedding"](base_x)
            parts = {"embedding": z, "pose": pose}
        else:
            z = self.encoder["embedding"](x)
            parts = {"embedding": z}

        return parts

    def decode(self, z, return_corrected=False):
        base_xhat = self.decoder[self.hparams.x_label](z["embedding"])
        clip_min = self.hparams.get("clip_min")
        clip_max = self.hparams.get("clip_min")

        if clip_min is not None or clip_max is not None:
            base_xhat = base_xhat.clip(clip_min, clip_max)

        if self.mask is not None:
            base_xhat = self.mask(base_xhat)

        if return_corrected:
            if self.hparams.group is not None:
                xhat = self.rotation_module(base_xhat, None, R=z["pose"])
            else:
                xhat = base_xhat
            return {self.hparams.x_label: base_xhat, "corrected": xhat}

        return {self.hparams.x_label: base_xhat}

    def calculate_elbo(self, x, xhat, z):
        with torch.no_grad():
            base_x = {
                self.hparams.x_label: self.rotation_module(
                    x[self.hparams.x_label], None, R=z["pose"].transpose(1, 2)
                )
            }

        return super().calculate_elbo(base_x, xhat, z)
