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
        rank: int = 0,
        n_epochs_before_dmd: int = 10,
        linearity_loss_factor: float = 1.0,
        Areg_factor: float = 1.0,
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
            "train/loss/Areg": MeanMetric(),
            "val/loss/Areg": MeanMetric(),
            "test/loss/Areg": MeanMetric(),
        }

        metrics = base_kwargs.pop("metrics", _DEFAULT_METRICS)

        super().__init__(**base_kwargs, metrics=metrics)

        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder
        self.rank = rank if rank not in (0, None) else latent_dim

    def forward(self, batch):
        return self.encoder(batch[self.hparams.x_label])

    def _dmd_linearize(self, z, z_prime, rank):
        """Adapted from dlkoopman."""

        # U: (latent_dim, min(B-1, latent_dim))
        # S: (min(B-1, latent_dim),)
        # V: (B-1, min(B-1, latent_dim))

        U, S, V = SVD.apply(z)

        if rank <= 0:
            rank = self._optimal_rank(S.detach().cpu().numpy(), z.shape[1])
        U = U[:, :rank]  # (latent_dim, rank)
        Ut = U.T  # (rank, latent_dim)

        S = torch.diag(S[:rank])  # shape = (rank, rank)

        # Right singular vectors
        V = V[:, :rank]  # shape = (B-1, rank)

        # Outputs
        intermediate = z_prime @ (V @ torch.linalg.inv(S))  # shape = (latent_dim, rank)
        Atilde = Ut @ intermediate  # shape = (rank, rank)

        Lambda, eigvecstilde = torch.linalg.eig(Atilde)
        log_Lambda = torch.diag(torch.log(Lambda))  # shape = (rank, rank)

        eigvecs = intermediate.type_as(eigvecstilde) @ eigvecstilde  # shape = (latent_dim, rank)

        return Atilde, log_Lambda, eigvecs

    def _dmd_predict(self, t, z0, log_Lambda, eigvecs):
        """Adapted from dlkoopman."""

        # coeffs = eigvecs.pinv(rtol=0.01) @ z0.type_as(eigvecs)
        coeffs = torch.linalg.lstsq(eigvecs, z0.type_as(eigvecs), rcond=0.01).solution

        log_Lambda_exp = torch.linalg.matrix_exp(
            log_Lambda.expand(len(t), log_Lambda.shape[0], log_Lambda.shape[1])
            * torch.tensor(t).view(len(t), 1, 1).type_as(log_Lambda)
        )

        z_pred = (eigvecs @ log_Lambda_exp) @ coeffs

        return z_pred.abs()  # norm of complex value

    def _optimal_rank(self, S, n_samples):
        beta = np.divide(*sorted((self.latent_dim, n_samples)))
        tau = np.median(S) * _omega(beta)
        rank = np.sum(S > tau)
        if rank == 0:
            warnings.warn(
                "SVD optimal rank is 0. The largest singular values are "
                "indistinguishable from noise. Setting rank truncation to 1.",
                RuntimeWarning,
            )
            rank = 1
        return rank

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def model_step(self, stage, batch, batch_idx):
        x = batch[self.hparams.x_label]

        z = self.encode(x)
        xhat = self.decode(z)

        recon_loss = _reduce_loss(F.mse_loss(xhat, x, reduction="none"))
        if self.current_epoch + 1 > self.hparams.n_epochs_before_dmd:
            rank = min(self.rank, len(z) - 1) if self.rank <= 0 else -1

            Atilde, log_Lambda, eigvecs = self._dmd_linearize(z[:-1].T, z[1:].T, rank)

            t = range(len(z) - 1)
            z_prime_hat = self._dmd_predict(t, z[0], log_Lambda, eigvecs)
            x_prime_hat = self.decoder(z_prime_hat)

            lin_loss = _reduce_loss(F.mse_loss(z_prime_hat, z[1:], reduction="none"))
            pred_loss = _reduce_loss(F.mse_loss(x_prime_hat, x[1:], reduction="none"))

            Areg = Atilde.abs().sum() / Atilde.numel()
        else:
            lin_loss = 0.0
            pred_loss = 0.0
            Areg = 0.0

        loss = {
            "loss": (
                recon_loss
                + self.hparams.linearity_loss_factor * lin_loss
                + pred_loss
                + self.hparams.Areg_factor * Areg
            ),
            "reconstruction": recon_loss,
            "linearity": lin_loss,
            "prediction": pred_loss,
            "Areg": Areg,
        }

        return loss, None, None
