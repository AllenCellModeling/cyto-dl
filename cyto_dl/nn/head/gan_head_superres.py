from typing import Callable

import numpy as np
import torch

from cyto_dl.models.im2im.utils.postprocessing import detach
from cyto_dl.nn.losses import Pix2PixHD

from .gan_head import GANHead
from .res_blocks_head import ResBlocksHead


class GANHead_resize(GANHead, ResBlocksHead):
    """Inherit run_head from GANHead, use __init__ and forward of ResBlocksHead."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gan_loss=Pix2PixHD(scales=1),
        reconstruction_loss=torch.nn.MSELoss(),
        reconstruction_loss_weight=100,
        postprocess={"input": detach, "prediction": detach},
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
        """
        ResBlocksHead.__init__(
            self,
            loss=None,
            in_channels=in_channels,
            out_channels=out_channels,
            final_act=final_act,
            postprocess=postprocess,
            resolution=resolution,
            spatial_dims=spatial_dims,
            n_convs=n_convs,
            dropout=dropout,
            upsample_method=upsample_method,
            upsample_ratio=upsample_ratio,
            first_layer=first_layer,
            dense=dense,
        )
        self.gan_loss = gan_loss
        self.reconstruction_loss = reconstruction_loss
        self.reconstruction_loss_weight = reconstruction_loss_weight

    def _ensure_same_shape(self, x, y):
        min_shape = np.minimum(x.shape, y.shape)
        x = x[:, :, : min_shape[2], : min_shape[3], : min_shape[4]]
        y = y[:, :, : min_shape[2], : min_shape[3], : min_shape[4]]
        return x, y

    def _calculate_loss(self, y_hat, batch, discriminator):
        batch[self.head_name], y_hat = self._ensure_same_shape(batch[self.head_name], y_hat)
        return GANHead._calculate_loss(self, y_hat, batch, discriminator)

    def forward(self, x):
        return ResBlocksHead.forward(self, x)
