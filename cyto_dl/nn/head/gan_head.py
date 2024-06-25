import math

import numpy as np
import torch

from cyto_dl.models.im2im.utils.postprocessing import detach
from cyto_dl.nn.losses import Pix2PixHD

from .base_head import BaseHead


class GANHead(BaseHead):
    """GAN Task head."""

    def __init__(
        self,
        gan_loss=Pix2PixHD(scales=1),
        reconstruction_loss=torch.nn.MSELoss(),
        reconstruction_loss_weight=100,
        postprocess={"input": detach, "prediction": detach},
        save_input=False,
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
        save_input=False
            Whether to save out example input images during training
        """
        super().__init__(None, postprocess, save_input)
        self.gan_loss = gan_loss
        self.reconstruction_loss = reconstruction_loss
        self.reconstruction_loss_weight = reconstruction_loss_weight

    def _calculate_loss(self, y_hat, batch, discriminator):
        # extract intermediate activations from discriminator for real and predicted images
        features_discriminator = discriminator(
            batch[self.x_key], batch[self.head_name], y_hat.detach()
        )
        loss_D = self.gan_loss(features_discriminator, "discriminator")

        # passability of generated images
        features_generator = discriminator(batch[self.x_key], batch[self.head_name], y_hat)
        loss_G = self.gan_loss(features_generator, "generator")
        # image reconstruction quality
        loss_reconstruction = self.reconstruction_loss(batch[self.head_name], y_hat)
        return loss_D, loss_G + loss_reconstruction * self.reconstruction_loss_weight

    def forward(self, x):
        return torch.nn.Tanh()(x)

    def run_head(
        self,
        backbone_features,
        batch,
        stage,
        save_image,
        discriminator=None,
        run_forward=True,
        y_hat=None,
    ):
        if run_forward:
            y_hat = self.forward(backbone_features)
        if y_hat is None:
            raise ValueError(
                "y_hat must be provided, either by passing it in or setting `run_forward=True`"
            )
        loss_D, loss_G = None, None
        if stage != "predict":
            if discriminator is None:
                raise ValueError(
                    "Discriminator must be specified for train, test, and validation steps."
                )
            loss_D, loss_G = self._calculate_loss(y_hat, batch, discriminator)

        y_hat_out, y_out = None, None
        if save_image:
            y_hat_out, y_out = self.save_image(y_hat, batch, stage)

        return {
            "loss_D": loss_D,
            "loss_G": loss_G,
            "y_hat_out": y_hat_out,
            "y_out": y_out,
        }
