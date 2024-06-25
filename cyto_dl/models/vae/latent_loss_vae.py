from typing import Optional, Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn.modules.loss import _Loss as Loss
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torchmetrics import MeanMetric

from .base_vae import BaseVAE


class LatentLossVAE(BaseVAE):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        x_label: str,
        x_dim: str,
        output_dim: str,
        continuous_labels: list,
        discrete_labels: list,
        argmax_discrete: bool,
        latent_loss: dict,
        latent_loss_target: dict,
        latent_loss_weights: dict,
        latent_loss_backprop_when: dict = None,
        prior: dict = None,
        latent_loss_optimizer: torch.optim.Optimizer = torch.optim.Adam,
        latent_loss_scheduler: LRScheduler = torch.optim.lr_scheduler.StepLR,
        beta: float = 1.0,
        get_rotation: bool = True,
        condition_encoder: Optional[dict] = None,
        condition_decoder: Optional[dict] = None,
        basal_head: Optional[dict] = None,
        basal_head_loss: Optional[dict] = None,
        basal_head_weight: Optional[dict] = None,
        **base_kwargs,
    ):
        metric_keys = [
            "train/loss",
            "val/loss",
            "test/loss",
            "train/loss/total_reconstruction",
            "val/loss/total_reconstruction",
            "test/loss/total_reconstruction",
            "train/loss/total_kld",
            "val/loss/total_kld",
            "test/loss/total_kld",
        ]

        extend_list = []
        for key in discrete_labels:
            extend_list = [
                f"train/loss/adv_{key}",
                f"val/loss/adv_{key}",
                f"test/loss/adv_{key}",
            ]
        extend_list = [item for sublist in extend_list for item in sublist]
        metric_keys.extend(extend_list)

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            x_label=x_label,
            beta=beta,
            prior=prior,
            metric_keys=metric_keys,
            **base_kwargs,
        )
        self.get_rotation = get_rotation
        self.continuous_labels = continuous_labels
        self.discrete_labels = discrete_labels
        if len(self.continuous_labels) > 1:
            self.comb_label = self.continuous_labels[-1] + f"_{self.continuous_labels[0]}"
        self.argmax_discrete = argmax_discrete

        if not isinstance(latent_loss, (dict, DictConfig)):
            assert x_label is not None
            latent_loss = {x_label: latent_loss}
        self.latent_loss = latent_loss
        self.latent_loss = nn.ModuleDict(latent_loss)

        if not isinstance(latent_loss_target, (dict, DictConfig)):
            assert x_label is not None
            latent_loss_target = {x_label: latent_loss_target}
        self.latent_loss_target = latent_loss_target

        if not isinstance(latent_loss_weights, (dict, DictConfig)):
            assert x_label is not None
            latent_loss_weights = {x_label: latent_loss_weights}
        self.latent_loss_weights = latent_loss_weights

        if not isinstance(latent_loss_backprop_when, (dict, DictConfig)):
            assert x_label is not None
            latent_loss_backprop_when = {x_label: latent_loss_backprop_when}
        self.latent_loss_backprop_when = latent_loss_backprop_when

        if not isinstance(latent_loss_optimizer, (dict, DictConfig)):
            assert x_label is not None
            latent_loss_optimizer = {x_label: latent_loss_optimizer}
        self.latent_loss_optimizer = latent_loss_optimizer

        if not isinstance(latent_loss_optimizer, (dict, DictConfig)):
            assert x_label is not None
            latent_loss_scheduler = {x_label: latent_loss_scheduler}
        self.latent_loss_scheduler = latent_loss_scheduler

        self.automatic_optimization = False
        self.condition_encoder = nn.ModuleDict(condition_encoder)
        self.condition_decoder = nn.ModuleDict(condition_decoder)
        self.basal_head = nn.ModuleDict(basal_head)
        self.basal_head_loss = nn.ModuleDict(basal_head_loss)
        self.basal_head_weight = basal_head_weight

    def parse_batch(self, batch):
        for i in self.discrete_labels:
            if len(batch[i].shape) != 2:
                batch[i] = batch[i].view(-1, 1)
        return batch

    def encode(self, batch, **kwargs):
        encoded = {}
        encoded = self.encoder[self.hparams.x_label](batch[self.hparams.x_label], **kwargs)
        for part in self.discrete_labels:
            if self.argmax_discrete:
                encoded[part] = self.encoder[part](batch[part].argmax(1))
            else:
                encoded[part] = self.encoder[part](batch[part])
            if len(encoded[part].shape) > len(encoded[self.hparams.x_label].shape):
                encoded[part] = encoded[part].squeeze(dim=1)

        if len(self.continuous_labels) > 1:
            encoded[self.continuous_labels[-1]] = (
                self.encoder[self.continuous_labels[0]](batch[self.comb_label])
                @ self.encoder[self.continuous_labels[-1]].weight
            )
        if self.basal_head:
            for key in self.basal_head.keys():
                encoded[key] = self.basal_head[key](encoded[self.hparams.x_label])
        return encoded

    def calculate_rcl_dict(self, x, xhat, z):
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

        if self.basal_head_loss:
            for key in self.basal_head_loss.keys():
                rcl_reduced[key] = self.basal_head_weight[key] * self.basal_head_loss[key](
                    z[key], x[key].squeeze(dim=-1)
                )
        return rcl_reduced

    def decode(self, z_parts, return_canonical=False, batch=None):
        if hasattr(self.encoder[self.hparams.x_label], "generate_grid_feats"):
            if self.encoder[self.hparams.x_label].generate_grid_feats:
                base_xhat = self.decoder[self.hparams.x_label](
                    batch[self.point_label], z_parts["grid_feats"]
                )
            else:
                base_xhat = self.decoder[self.hparams.x_label](z_parts[self.hparams.x_label])
        else:
            base_xhat = self.decoder[self.hparams.x_label](z_parts[self.hparams.x_label])

        if self.get_rotation:
            rotation = z_parts["rotation"]
            xhat = torch.einsum("bij,bjk->bik", base_xhat[:, :, :3], rotation)
            if xhat.shape[-1] != base_xhat.shape[-1]:
                xhat = torch.cat([xhat, base_xhat[:, :, -1:]], dim=-1)
        else:
            xhat = base_xhat

        if return_canonical:
            return {self.hparams.x_label: xhat, "canonical": base_xhat}

        return {self.hparams.x_label: xhat}

    def forward(self, batch, decode=False, inference=True, return_params=False, **kwargs):
        is_inference = inference or not self.training

        z_params = self.encode(batch, get_rotation=self.get_rotation)
        z_composed = {}
        z_composed[self.hparams.x_label] = self.latent_compose_function(z_params)
        for key in z_params.keys():
            if key != self.hparams.x_label:
                z_composed[key] = z_params[key]

        z = self.sample_z(z_composed, inference=inference)
        z = self.decoder_compose_function(z, batch)

        if not decode:
            return z

        xhat = self.decode(z)
        if return_params:
            return xhat, z, z_params, z_composed

        return xhat, z

    def latent_compose_function(self, z_parts, **kwargs):
        if self.discrete_labels:
            for j, key in enumerate([self.hparams.x_label] + self.discrete_labels):
                this_z_parts = z_parts[key]
                if len(this_z_parts.shape) == 3:
                    # this_z_parts = torch.squeeze(z_parts[key], dim=(-1))
                    this_z_parts = this_z_parts.argmax(dim=1)
                if j == 0:
                    cond_feats = this_z_parts
                else:
                    cond_feats = torch.cat((cond_feats, this_z_parts), dim=1)
            # shared encoder
            z_parts[self.hparams.x_label] = self.condition_encoder[self.hparams.x_label](
                cond_feats
            )

        return z_parts[self.hparams.x_label]

    def decoder_compose_function(self, z_parts, batch):
        # import ipdb
        # ipdb.set_trace()
        if self.discrete_labels:
            for j, key in enumerate(self.discrete_labels):
                if j == 0:
                    cond_inputs = batch[key]
                    # cond_inputs = torch.squeeze(batch[key], dim=(-1))
                else:
                    cond_inputs = torch.cat((cond_inputs, batch[key]), dim=1)
                cond_feats = torch.cat((cond_inputs, z_parts[self.hparams.x_label]), dim=1)
            # shared decoder
            z_parts[self.hparams.x_label] = self.condition_decoder[self.hparams.x_label](
                cond_feats
            )
        return z_parts

    def model_step(self, stage, batch, batch_idx):
        (xhat, z_parts, z_parts_params, z_composed) = self.forward(
            batch, decode=True, inference=False, return_params=True
        )

        (
            loss,
            reconstruction_loss,
            rec_loss_per_part,
            kld_loss,
            kld_per_part,
        ) = self.calculate_elbo(batch, xhat, z_composed)

        # if stage == 'train':
        #     print(reconstruction_loss, z_composed.max(), batch['drug_dose'].unique())

        mu = z_parts_params[self.hparams.x_label]
        if mu.shape[1] != self.latent_dim:
            mu = mu[:, : int(mu.shape[1] / 2)]

        _loss = {}
        _adv_preds = {}

        weighted_adv_loss = 0
        adv_loss = 0
        for part in self.latent_loss.keys():
            if isinstance(self.latent_loss[part].loss, torch.nn.modules.loss.BCEWithLogitsLoss):
                batch[self.latent_loss_target[part]] = batch[self.latent_loss_target[part]].gt(0)

            _loss[part], _adv_preds[part] = self.latent_loss[part](
                mu, batch[self.latent_loss_target[part]], return_pred=True
            )
            adv_loss += _loss[part]
            this_weight = self.latent_loss_weights.get(part, 1.0)
            weighted_adv_loss += _loss[part] * this_weight

        def compute_gradients(output, input):
            grads = torch.autograd.grad(output, input, create_graph=True)
            grads = grads[0].pow(2).mean()
            return grads

        if stage != "test":
            main_optim, opt_adv = self.optimizers()
            lr_schedulers = self.lr_schedulers()

            adversarial_flag = False
            for optim_ix, optim in enumerate([opt_adv]):
                group_key = self.latent_loss_optimizer_map[optim_ix]
                mod = self.latent_loss_backprop_when.get(group_key) or 3

                if stage == "train":
                    # print(self.global_step)
                    if (batch_idx % mod == 0) & (self.current_epoch > 40):
                        adversarial_flag = True

                        adversary_penalty = compute_gradients(
                            _adv_preds[self.discrete_labels[0]].sum(), mu
                        )

                        optim.zero_grad()
                        self.manual_backward(adv_loss + this_weight * adversary_penalty)
                        optim.step()
                        # Dont use LR scheduler here, messes up the loss
                        # if lr_sched is not None:
                        #     lr_sched.step()

            if (stage == "train") and (not adversarial_flag):
                main_optim.zero_grad()

                self.manual_backward(
                    reconstruction_loss + self.beta * kld_loss - weighted_adv_loss
                )
                main_optim.step()

        total_loss = reconstruction_loss + self.beta * kld_loss - weighted_adv_loss

        loss = {
            "loss": total_loss,
            "total_kld": kld_loss.detach(),
            "total_reconstruction": reconstruction_loss.detach(),
        }

        for part, recon_part in rec_loss_per_part.items():
            loss[f"reconstruction_{part}"] = recon_part.detach()

        for part, z_part in z_parts.items():
            if not isinstance(z_part, dict):
                loss[f"z/{part}"] = z_part.detach()
                loss[f"z_params/{part}"] = z_parts_params[part].detach()

        for part, z_part in z_composed.items():
            if not isinstance(z_part, dict):
                loss[f"z_composed/{part}"] = z_part.detach()

        for part in self.prior:
            loss[f"kld_{part}"] = kld_per_part[part].detach()

        for part, value in _loss.items():
            loss[f"adv_{part}"] = value.detach()

        return loss, None, None

    def get_params(self):
        return list(self.parameters())

    def configure_optimizers(self):
        # get_params = lambda self: list(self.parameters())

        _parameters = self.get_params(self.decoder[self.hparams.x_label])

        for part in self.optimizer["main"]["keys"]:
            _parameters.extend(self.get_params(self.encoder[part]))

        optimizers = [self.optimizer["main"]["opt"](_parameters)]

        lr_schedulers = [
            (
                self.lr_scheduler["main"](optimizer=optimizers[0])
                if self.lr_scheduler["main"] is not None
                else None
            )
        ]

        self.latent_loss_optimizer_map = {}
        for optim_ix, (group_key, group) in enumerate(self.latent_loss_optimizer.items()):
            self.latent_loss_optimizer_map[optim_ix] = group_key
            _parameters3 = self.get_params(self.latent_loss[group["keys"][0]])
            if len(group["keys"]) > 1:
                for key in group["keys"][1:]:
                    _parameters3.extend(self.get_params(self.latent_loss[key]))
            optimizers.append(group["opt"](_parameters3))
            lr_schedulers.append(self.latent_loss_scheduler[group_key](optimizer=optimizers[-1]))

        return optimizers, lr_schedulers
