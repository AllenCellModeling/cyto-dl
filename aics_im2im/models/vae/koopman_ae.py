import warnings
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanMetric

from aics_im2im.models.base_model import BaseModel
from aics_im2im.utils.svd import SVD


def _omega(x):
    return 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43


def _reduce_loss(loss):
    return loss.reshape(loss.shape[0], -1).mean(dim=1).mean()


class KoopmanAE(BaseModel):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        x_label: str,
        n_epochs_before_koopman: int = 10,
        linearity_loss_factor: float = 1.0,
        K_l2_reg: float = 0.01,
        **base_kwargs,
    ):
        """Instantiate a Koopman AE model.
        Parameters
        ----------
        encoder: nn.Module
            Encoder network
        decoder: nn.Module
            Decoder network
        latent_dim: int
            Bottleneck size
        x_label: Optional[str] = None
        **base_kwargs:
            Additional arguments passed to BaseModel
        """

        _DEFAULT_METRICS = {
            "train/loss": MeanMetric(),
            "val/loss": MeanMetric(),
            "test/loss": MeanMetric(),
            "train/loss/reconstruction": MeanMetric(),
            "val/loss/reconstruction": MeanMetric(),
            "test/loss/reconstruction": MeanMetric(),
            "train/loss/prediction": MeanMetric(),
            "val/loss/prediction": MeanMetric(),
            "test/loss/prediction": MeanMetric(),
            "train/loss/linearity": MeanMetric(),
            "val/loss/linearity": MeanMetric(),
            "test/loss/linearity": MeanMetric(),
        }

        metrics = base_kwargs.pop("metrics", _DEFAULT_METRICS)

        super().__init__(**base_kwargs, metrics=metrics)

        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)

    def configure_optimizers(self):
        optimizer = self.optimizer(
            [
                {"params": self.K.parameters(), "weight_decay": self.hparams.K_l2_reg},
                {"params": self.encoder.parameters(), "weight_decay": 0},
                {"params": self.decoder.parameters(), "weight_decay": 0},
            ]
        )

        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "frequency": 1,
                },
            }
        return optimizer

    def forward(self, batch):
        return self.encoder(batch[self.hparams.x_label])

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def model_step(self, stage, batch, batch_idx):
        x = batch[self.hparams.x_label]

        z = self.encode(x)
        xhat = self.decode(z)

        recon_loss = _reduce_loss(F.mse_loss(xhat, x, reduction="none"))
        if self.current_epoch + 1 > self.hparams.n_epochs_before_koopman:
            z_prime_hat = self.K(z[:1])
            x_prime_hat = self.decoder(z_prime_hat)

            lin_loss = _reduce_loss(F.mse_loss(z_prime_hat, z[1:], reduction="none"))
            pred_loss = _reduce_loss(F.mse_loss(x_prime_hat, x[1:], reduction="none"))

        else:
            lin_loss = 0.0
            pred_loss = 0.0

        loss = {
            "loss": (recon_loss + self.hparams.linearity_loss_factor * lin_loss + pred_loss),
            "reconstruction": recon_loss,
            "linearity": lin_loss,
            "prediction": pred_loss,
        }

        return loss, None, None
