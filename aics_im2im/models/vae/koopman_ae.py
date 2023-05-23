from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss as Loss
from torchmetrics import MeanMetric

from aics_im2im.models.base_model import BaseModel


class KoopmanAE(BaseModel):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        x_label: str,
        rank: int = 0,
        reconstruction_loss: Loss = nn.MSELoss(reduction="mean"),
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
        reconstruction_loss: Loss
            Loss to be used for reconstruction. Can be a PyTorch loss or a class
            that respects the same interface,
            i.e. subclasses torch.nn.modules._Loss
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

        super().__init__(**base_kwargs)

        self.reconstruction_loss = reconstruction_loss
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder
        self.rank = rank if rank not in (0, None) else latent_dim

    def forward(self, batch):
        return self.encoder(batch[self.hparams.x_label])

    def _dmd_linearize(self, z, z_prime, rank):
        """Adapted from dlkoopman."""

        # U: (latent_dim, min(B-1, latent_dim))
        # Sigma: (min(B-1, latent_dim),)
        # V: (B-1, min(B-1, latent_dim))

        U, Sigma, V = torch.linalg.svd(z)

        U = U[:, :rank]  # (latent_dim, rank)
        Ut = U.T  # (rank, latent_dim)

        Sigma = torch.diag(Sigma[:rank])  # shape = (rank, rank)

        # Right singular vectors
        V = V[:, :rank]  # shape = (B-1, rank)

        # Outputs
        intermediate = z_prime @ (V @ torch.linalg.inv(Sigma))  # shape = (latent_dim, rank)
        Atilde = Ut @ intermediate  # shape = (rank, rank)

        Lambda, eigvecstilde = torch.linalg.eig(Atilde)
        logLambda = torch.diag(torch.log(Lambda))  # shape = (rank, rank)

        eigvecs = intermediate.type_as(eigvecstilde) @ eigvecstilde  # shape = (latent_dim, rank)

        return Atilde, logLambda, eigvecs

    def _dmd_predict(self, t, z0, logLambda, eigvecs):
        """Adapted from dlkoopman."""

        coeffs = torch.linalg.pinv(eigvecs, rtol=0.01) @ z0.type_as(eigvecs)

        logLambda_exp = torch.linalg.matrix_exp(
            logLambda.expand(len(t), logLambda.shape[0], logLambda.shape[1])
            * torch.tensor(t).view(len(t), 1, 1).type_as(logLambda)
        )

        z_pred = (logLambda_exp @ coeffs) @ eigvecs.T

        return z_pred.abs()  # norm of complex value

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def model_step(self, stage, batch, batch_idx):
        x = batch[self.hparams.x_label]

        z = self.encode(x)
        xhat = self.decode(z)

        rank = min(self.rank, len(z) - 1)
        Atilde, logLambda, eigvecs = self._dmd_linearize(z[:-1].T, z[1:].T, rank)

        t = range(len(z) - 1)
        z_prime_hat = self._dmd_predict(t, z[0], logLambda, eigvecs)
        x_prime_hat = self.decoder(z_prime_hat)

        recon_loss = self.reconstruction_loss(xhat, x)
        lin_loss = F.mse_loss(z_prime_hat, z[1:], reduce="mean")
        pred_loss = self.reconstruction_loss(x_prime_hat, x[1:])

        loss = {
            "loss": (recon_loss + lin_loss + pred_loss).mean(),
            "loss/reconstruction": recon_loss,
            "loss/linearity": lin_loss,
            "loss/prediction": pred_loss,
        }

        return loss, None, None
