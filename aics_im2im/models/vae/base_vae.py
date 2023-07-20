from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn.modules.loss import _Loss as Loss
from torchmetrics import MeanMetric

from aics_im2im.models.base_model import BaseModel

from .priors import IdentityPrior, IsotropicGaussianPrior, Prior


class BaseVAE(BaseModel):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        x_label: str,
        beta: float = 1.0,
        id_label: Optional[str] = None,
        reconstruction_loss: Loss = nn.MSELoss(reduction="none"),
        prior: Optional[Sequence[Prior]] = None,
        decoder_latent_parts: Optional[Dict[str, Sequence[str]]] = None,
        **base_kwargs,
    ):
        """Instantiate a basic VAE model.
        Parameters
        ----------
        encoder: nn.Module
            Encoder network
        decoder: nn.Module
            Decoder network
        x_label: Optional[str] = None
        id_label: Optional[str] = None
        beta: float = 1.0
            Beta parameter - the weight of the KLD term in the loss function
        reconstruction_loss: Loss
            Loss to be used for reconstruction. Can be a PyTorch loss or a class
            that respects the same interface,
            i.e. subclasses torch.nn.modules._Loss
        prior: Optional[Sequence[AbstractPrior]]
            List of prior specifications to use for latent space
        decoder_latent_parts: Optional[Dict[str, Sequence[str]]] = None
            Dictionary that specifies for each output part's decoer, what latent
            keys it depends on
        **base_kwargs:
            Additional arguments passed to BaseModel
        """

        _DEFAULT_METRICS = {
            "train/loss": MeanMetric(),
            "val/loss": MeanMetric(),
            "test/loss": MeanMetric(),
            "train/loss/total_reconstruction": MeanMetric(),
            "val/loss/total_reconstruction": MeanMetric(),
            "test/loss/total_reconstruction": MeanMetric(),
            "train/loss/total_kld": MeanMetric(),
            "val/loss/total_kld": MeanMetric(),
            "test/loss/total_kld": MeanMetric(),
        }

        if not isinstance(prior, (dict, DictConfig)):
            prior = {"embedding": prior}

        for part in prior.keys():
            _DEFAULT_METRICS.update(
                {
                    f"train/loss/kld_{part}": MeanMetric(),
                    f"val/loss/kld_{part}": MeanMetric(),
                    f"test/loss/kld_{part}": MeanMetric(),
                }
            )

        if not isinstance(reconstruction_loss, (dict, DictConfig)):
            assert x_label is not None
            recon_parts = [x_label]
        else:
            recon_parts = reconstruction_loss.keys()

        for part in recon_parts:
            _DEFAULT_METRICS.update(
                {
                    f"train/loss/reconstruction_{part}": MeanMetric(),
                    f"val/loss/reconstruction_{part}": MeanMetric(),
                    f"test/loss/reconstruction_{part}": MeanMetric(),
                }
            )

        metrics = base_kwargs.pop("metrics", _DEFAULT_METRICS)

        super().__init__(metrics=metrics, **base_kwargs)

        for key in prior.keys():
            if isinstance(prior[key], (str, type(None))):
                if prior[key] == "gaussian":
                    prior[key] = IsotropicGaussianPrior(dimensionality=latent_dim)
                else:
                    prior[key] = IdentityPrior(dimensionality=latent_dim)
            elif not isinstance(prior[key], Prior):
                raise ValueError(
                    f"Expected prior to either be one of ('gaussian', 'identity', None)"
                    f"or an object of type `Prior`. Got: {type(prior)}"
                )
        self.prior = nn.ModuleDict(prior)

        self.reconstruction_loss = reconstruction_loss

        if not isinstance(encoder, (dict, DictConfig)):
            encoder = {"embedding": encoder}
        self.encoder = nn.ModuleDict(encoder)

        if not isinstance(decoder, (dict, DictConfig)):
            assert x_label is not None
            decoder = {x_label: decoder}
        self.decoder = nn.ModuleDict(decoder)

        if not isinstance(reconstruction_loss, (dict, DictConfig)):
            assert x_label is not None
            reconstruction_loss = {x_label: reconstruction_loss}
        self.reconstruction_loss = nn.ModuleDict(reconstruction_loss)

        self.beta = beta
        self.latent_dim = latent_dim

        if decoder_latent_parts is None:
            self.decoder_latent_parts = {
                key: self.prior.keys() for key in self.decoder.keys()
            }
        else:
            self.decoder_latent_parts = decoder_latent_parts
            for key in self.decoder.keys():
                if key not in self.decoder_latent_parts:
                    raise KeyError(
                        f"Decoder with key '{key}' doesn't have an entry in "
                        f"`decoder_latent_parts`, so we don't know which "
                        "latent parts it uses."
                    )

    def calculate_rcl(self, x, xhat, key):
        rcl_per_input_dimension = self.reconstruction_loss[key](x[key], xhat[key])
        return rcl_per_input_dimension

    def calculate_elbo(self, x, xhat, z):
        rcl_per_input_dimension = {}
        rcl_reduced = {}
        for key in xhat.keys():
            rcl_per_input_dimension[key] = self.calculate_rcl(x, xhat, key)
            if len(rcl_per_input_dimension[key].shape) > 0:
                rcl = (
                    rcl_per_input_dimension[key]
                    # flatten
                    .view(rcl_per_input_dimension[key].shape[0], -1)
                    # and sum across each batch element's dimensions
                    .sum(dim=1)
                )

                rcl_reduced[key] = rcl.mean()
            else:
                rcl_reduced[key] = rcl_per_input_dimension[key]

        kld_per_part = {
            part: prior(z[part], mode="kl", reduction="none")
            for part, prior in self.prior.items()
        }

        kld_per_part_summed = {
            part: kl.sum(dim=-1).mean() for part, kl in kld_per_part.items()
        }

        total_kld = sum(kld_per_part_summed.values())
        total_recon = sum(rcl_reduced.values())
        return (
            total_recon + self.beta * total_kld,
            total_recon,
            rcl_reduced,
            total_kld,
            kld_per_part,
        )

    def sample_z(self, z_parts_params, inference=False):
        z = {}
        for part, part_params in z_parts_params.items():
            if part in self.prior:
                z[part] = self.prior[part](
                    part_params, mode="sample", inference=inference
                )
            else:
                # if prior for this part isn't in the dict, assume dirac prior
                # i.e. just return the params, and it won't contribute to kl
                z[part] = part_params

        return z

    def encode(self, batch):
        return {part: encoder(batch[part]) for part, encoder in self.encoder.items()}

    def decode(self, z):
        # for each decoder key, get the latent parts it uses from `self.decoder_latent_keys`
        # and pass them as *args to that decoder's forward method
        return {
            part: decoder(*[z[key] for key in self.decoder_latent_keys[part]])
            for part, decoder in self.decoder.items()
        }

    def forward(self, batch, decode=False, inference=True, return_params=False):
        is_inference = inference or not self.training

        z_params = self.encode(batch)
        z = self.sample_z(z_params, inference=inference)

        if not decode:
            return z

        xhat = self.decode(z)
        if return_params:
            return xhat, z, z_params

        return xhat, z

    def model_step(self, stage, batch, batch_idx):
        (
            xhat,
            z,
            z_params,
        ) = self.forward(batch, decode=True, inference=False, return_params=True)

        (
            loss,
            rec_loss,
            rec_loss_per_part,
            kld_loss,
            kld_per_part,
        ) = self.calculate_elbo(batch, xhat, z_params)

        loss = {
            "loss": loss,
            "total_kld": kld_loss.detach(),
            "total_reconstruction": rec_loss.detach(),
        }

        for part, recon_part in rec_loss_per_part.items():
            loss[f"reconstruction_{part}"] = recon_part.detach()

        preds = {}
        for part, z_part in z.items():
            preds[f"z/{part}"] = z_part.detach()
            preds[f"z_params/{part}"] = z_params[part].detach()

        for part in self.prior:
            loss[f"kld_{part}"] = kld_per_part[part].detach()

        if self.hparams.id_label is not None:
            if self.hparams.id_label in batch:
                ids = batch[self.hparams.id_label].detach()
                preds.update({self.hparams.id_label: ids, "id": ids})

        return loss, preds, None
