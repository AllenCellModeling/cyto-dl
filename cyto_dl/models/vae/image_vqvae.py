import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from cyto_dl.models.vae.image_vae import ImageVAE
from monai.networks.layers.factories import Act, Norm
from .point_cloud_vqvae import VectorQuantizerEMA

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


class _Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = torch.tensor(scale)

    def forward(self, x):
        return x * self.scale.type_as(x)


class ImageVQVAE(ImageVAE):
    def __init__(
        self,
        x_label: str,
        latent_dim: int,
        spatial_dims: int,
        num_embeddings: int,
        commitment_cost: float,
        decay: float,
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
        encoder_padding: Optional[Union[int, Sequence[int]]] = None,
        eps: float = 1e-8,
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
            x_label=x_label,
            latent_dim=latent_dim,
            spatial_dims=spatial_dims,
            in_shape=in_shape,
            channels=channels,
            strides=strides,
            kernel_sizes=kernel_sizes,
            group=group,
            out_channels=out_channels,
            decoder_initial_shape=decoder_initial_shape,
            decoder_channels=decoder_channels,
            decoder_strides=decoder_strides,
            maximum_frequency=maximum_frequency,
            background_value=background_value,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            prior=prior,
            last_act=last_act,
            last_scale=last_scale,
            mask_input=mask_input,
            mask_output=mask_output,
            clip_min=clip_min,
            clip_max=clip_max,
            num_res_units=num_res_units,
            up_kernel_size=up_kernel_size,
            first_conv_padding_mode=first_conv_padding_mode,
            encoder_padding=encoder_padding,
            eps=eps,
            metric_keys=metric_keys,
            **base_kwargs,
        )
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.vq_layer = nn.ModuleDict(
            {
                self.x_label: VectorQuantizerEMA(
                    latent_dim, self.num_embeddings, self.commitment_cost, self.decay
                )
            }
        )

    def forward(
        self, batch, decode=False, inference=True, return_params=False, **kwargs
    ):
        is_inference = inference or not self.training

        z_params = self.encode(batch, **kwargs)
        quantized, commitment_loss = self.vq_layer[self.x_label](z_params["embedding"])
        z = z_params.copy()
        z["embedding"] = quantized

        if not decode:
            return quantized

        xhat = self.decode(z)
        if return_params:
            return xhat, z, z_params, commitment_loss

        return xhat, z

    def model_step(self, stage, batch, batch_idx):
        (xhat, z, z_params, commitment_loss) = self.forward(
            batch, decode=True, inference=False, return_params=True
        )

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
