import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import nn

from cyto_dl.models.vae.point_cloud_vae import PointCloudVAE
import torch.nn.functional as F

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False
from vector_quantize_pytorch import ResidualVQ, FSQ, VectorQuantize



class PointCloudVQVAE2(PointCloudVAE):
    def __init__(
        self,
        num_embeddings: int,
        commitment_cost: float,
        decay: float,
        latent_dim: int,
        x_label: str,
        encoder: dict,
        decoder: dict,
        reconstruction_loss: dict,
        prior: dict,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        get_rotation=False,
        beta: float = 1.0,
        point_label: Optional[str] = "points",
        occupancy_label: Optional[str] = "points.df",
        embedding_head: Optional[dict] = None,
        embedding_head_loss: Optional[dict] = None,
        embedding_head_weight: Optional[dict] = None,
        basal_head: Optional[dict] = None,
        basal_head_loss: Optional[dict] = None,
        basal_head_weight: Optional[dict] = None,
        condition_encoder: Optional[dict] = None,
        condition_decoder: Optional[dict] = None,
        condition_keys: Optional[list] = None,
        disable_metrics: Optional[bool] = False,
        **base_kwargs,
    ):

        metric_keys = [
            "train/loss",
            "val/loss",
            "test/loss",
            "train/loss/total_reconstruction",
            "val/loss/total_reconstruction",
            "test/loss/total_reconstruction",
        ]

        super().__init__(
            latent_dim=latent_dim,
            x_label=x_label,
            encoder=encoder,
            decoder=decoder,
            reconstruction_loss=reconstruction_loss,
            prior=prior,
            optimizer=optimizer,
            get_rotation=get_rotation,
            beta=beta,
            point_label=point_label,
            occupancy_label=occupancy_label,
            embedding_head=embedding_head,
            embedding_head_loss=embedding_head_loss,
            embedding_head_weight=embedding_head_weight,
            basal_head=basal_head,
            basal_head_los=basal_head_loss,
            basal_head_weight=basal_head_weight,
            condition_encoder=condition_encoder,
            condition_decoder=condition_decoder,
            condition_keys=condition_keys,
            disable_metrics=disable_metrics,
            metric_keys=metric_keys,
            **base_kwargs,
        )

        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay

        # self.vq_layer = nn.ModuleDict(
        #     {
        #         x_label: VectorQuantizerEMA(
        #             latent_dim, self.num_embeddings, self.commitment_cost, self.decay
        #         )
        #     }
        # )

        self.vq_layer = nn.ModuleDict(
            {
                x_label: VectorQuantize(
                    dim = 256,
                    codebook_size = 256,
                    codebook_dim = 16      # paper proposes setting this to 32 or as low as 8 to increase codebook usage
                )
            }
        )

    def forward(self, batch, decode=False, inference=True, return_params=False):
        is_inference = inference or not self.training

        z_params = self.encode(batch, get_rotation=self.get_rotation)
        z_params = self.encoder_compose_function(z_params)

        batch_size = batch[self.hparams.x_label].shape[0]
        q_in = z_params[self.hparams.x_label].unsqueeze(dim=1)
        # import ipdb
        # ipdb.set_trace()
        quantized, _, commitment_loss = self.vq_layer[self.hparams.x_label](q_in)
        z = z_params.copy()
        z[self.hparams.x_label] = quantized.view(batch_size, 256)

        z = self.decoder_compose_function(z, batch)

        if not decode:
            return z

        if hasattr(self.encoder[self.hparams.x_label], "generate_grid_feats"):
            if self.encoder[self.hparams.x_label].generate_grid_feats:
                xhat = self.decode(z, batch=batch)
            else:
                xhat = self.decode(z)
        else:
            xhat = self.decode(z)

        if return_params:
            return xhat, z, z_params, commitment_loss

        return xhat, z, commitment_loss

    def model_step(self, stage, batch, batch_idx):
        (
            xhat,
            z,
            z_params,
            commitment_loss
        ) = self.forward(batch, decode=True, inference=False, return_params=True)

        (
            loss,
            rec_loss,
            rec_loss_per_part,
            kld_loss,
            kld_per_part,
        ) = self.calculate_elbo(batch, xhat, z_params)

        loss = loss + commitment_loss

        loss = {
            "loss": loss,
            "total_kld": kld_loss.detach(),
            "total_reconstruction": rec_loss.detach(),
        }

        for part, recon_part in rec_loss_per_part.items():
            loss[f"reconstruction_{part}"] = recon_part.detach()

        preds = {}

        for part, z_part in z.items():
            if not isinstance(z_part, dict):
                preds[f"z/{part}"] = z_part.detach()
                preds[f"z_params/{part}"] = z_params[part].detach()

        return loss, preds, None


# EMA process
class ExponentialMovingAverage(nn.Module):
    """Maintains an exponential moving average for a value.

    This module keeps track of a hidden exponential moving average that is
    initialized as a vector of zeros which is then normalized to give the average.
    This gives us a moving average which isn't biased towards either zero or the
    initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)

    Initially:
        hidden_0 = 0
    Then iteratively:
        hidden_i = hidden_{i-1} - (hidden_{i-1} - value) * (1 - decay)
        average_i = hidden_i / (1 - decay^i)
    """

    def __init__(self, init_value, decay):
        super().__init__()

        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros_like(init_value))

    def forward(self, value):
        self.counter += 1
        self.hidden.sub_((self.hidden - value) * (1 - self.decay))
        average = self.hidden / (1 - self.decay**self.counter)
        return average


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class VectorQuantizerEMA(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized.
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
    """

    def __init__(
        self, embedding_dim, num_embeddings, commitment_cost, decay, epsilon=1e-5
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon

        # initialize embeddings as buffers
        embeddings = torch.empty(self.num_embeddings, self.embedding_dim)
        nn.init.xavier_uniform_(embeddings)
        self.register_buffer("embeddings", embeddings)
        self.ema_dw = ExponentialMovingAverage(self.embeddings, decay)

        # also maintain ema_cluster_size， which record the size of each embedding
        self.ema_cluster_size = ExponentialMovingAverage(
            torch.zeros((self.num_embeddings,)), decay
        )

    def forward(self, x):
        flat_x = x.reshape(-1, self.embedding_dim)

        # Use index to find embeddings in the latent space
        encoding_indices = self.get_code_indices(flat_x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x)

        # EMA
        with torch.no_grad():
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            updated_ema_cluster_size = self.ema_cluster_size(
                torch.sum(encodings, dim=0)
            )
            n = torch.sum(updated_ema_cluster_size)
            updated_ema_cluster_size = (
                (updated_ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            dw = torch.matmul(
                encodings.t(), flat_x
            )  # sum encoding vectors of each cluster
            updated_ema_dw = self.ema_dw(dw)
            normalised_updated_ema_w = (
                updated_ema_dw / updated_ema_cluster_size.reshape(-1, 1)
            )
            self.embeddings.data = normalised_updated_ema_w

        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()

        return quantized, loss

    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
            torch.sum(flat_x**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings**2, dim=1)
            - 2.0 * torch.matmul(flat_x, self.embeddings.t())
        )  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return F.embedding(encoding_indices, self.embeddings)
