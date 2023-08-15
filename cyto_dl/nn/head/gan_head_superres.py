import math
from typing import Callable

import numpy as np
import torch
from monai.networks.blocks import DenseBlock, UnetOutBlock, UnetResBlock, UpSample

from cyto_dl.models.im2im.utils.postprocessing import detach
from cyto_dl.nn.losses import Pix2PixHD

from .gan_head import GANHead


class GANHead_resize(GANHead):
    """GAN Task head with upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gan_loss=Pix2PixHD(scales=1),
        reconstruction_loss=torch.nn.MSELoss(),
        reconstruction_loss_weight=100,
        postprocess={"input": detach, "prediction": detach},
        calculate_metric=False,
        save_raw=False,
        final_act: Callable = torch.nn.Identity(),
        resolution="lr",
        spatial_dims=3,
        n_convs=1,
        dropout=0.0,
        upsample_method="pixelshuffle",
        upsample_ratio=None,
        first_layer=torch.nn.Identity(),
        dense: bool = False,
    ):
        """
        Parameters
        ----------
        gan_loss=Pix2PixHD(scales=1)
            Loss for optimizing GAN
        reconstruction_loss=torch.nn.MSELoss()
            Loss for optimizing generator's image reconstructions
        reconstruction_loss_weight=100
            Weighting of reconstruction loss
        postprocess={"input": detach, "prediction": detach}
            Postprocessing for `input` and `predictions` of head
        calculate_metric=False
            Whether to calculate a metric during training. Not used by GAN head.
        save_raw=False
            Whether to save out example input images during training
        """
        super().__init__(
            gan_loss,
            reconstruction_loss,
            reconstruction_loss_weight,
            postprocess,
            calculate_metric,
            save_raw,
        )

        self.resolution = resolution
        conv_input_channels = in_channels
        modules = [first_layer]
        upsample = torch.nn.Identity()

        upsample_ratio = upsample_ratio or [2] * spatial_dims

        if resolution == "hr":
            if upsample_method == "pixelshuffle":
                conv_input_channels //= 2**spatial_dims
            assert len(upsample_ratio) == spatial_dims
            upsample = UpSample(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=conv_input_channels,
                scale_factor=upsample_ratio,
                mode=upsample_method,
            )
        for i in range(n_convs):
            in_channels = conv_input_channels
            if dense:
                in_channels = (i + 1) * conv_input_channels
            modules.append(
                UnetResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=conv_input_channels,
                    stride=1,
                    kernel_size=3,
                    norm_name="INSTANCE",
                    dropout=dropout,
                )
            )
        if dense:
            # dense convolutions
            modules = [modules[0]] + [DenseBlock(modules[1:])]
            conv_input_channels *= n_convs + 1
        modules.extend(
            (
                UnetOutBlock(
                    spatial_dims=spatial_dims,
                    in_channels=conv_input_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                ),
                final_act,
            )
        )
        self.model = torch.nn.ModuleDict(
            {"upsample": upsample, "model": torch.nn.Sequential(*modules)}
        )

    def _ensure_same_shape(self, x, y):
        min_shape = np.minimum(x.shape, y.shape)
        x = x[:, :, : min_shape[2], : min_shape[3], : min_shape[4]]
        y = y[:, :, : min_shape[2], : min_shape[3], : min_shape[4]]
        return x, y

    def _calculate_loss(self, y_hat, batch, discriminator):
        # extract intermediate activations from discriminator for real and predicted images
        y, y_hat = self._ensure_same_shape(batch[self.head_name], y_hat)

        features_discriminator = discriminator(batch[self.x_key], y, y_hat.detach())
        loss_D = self.gan_loss(features_discriminator, "discriminator")

        # passability of generated images
        features_generator = discriminator(batch[self.x_key], y, y_hat)
        loss_G = self.gan_loss(features_generator, "generator")
        # image reconstruction quality
        loss_reconstruction = self.reconstruction_loss(y, y_hat)
        return loss_D, loss_G + loss_reconstruction * self.reconstruction_loss_weight

    def forward(self, x):
        if self.resolution == "hr":
            x = self.model["upsample"](x)
        return self.model["model"](x)
