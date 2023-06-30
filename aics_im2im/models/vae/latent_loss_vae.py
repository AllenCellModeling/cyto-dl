from typing import Optional, Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn.modules.loss import _Loss as Loss
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

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
        **base_kwargs,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            x_label=x_label,
            beta=beta,
            prior=prior,
            **base_kwargs,
        )

        self.continuous_labels = continuous_labels
        self.discrete_labels = discrete_labels
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

    def parse_batch(self, batch):
        for i in self.discrete_labels:
            if len(batch[i].shape) != 2:
                batch[i] = batch[i].view(-1, 1)
        return batch

    def encode(self, batch):
        encoded = {}
        encoded[self.hparams.x_label] = self.encoder[self.hparams.x_label](
            batch[self.hparams.x_label]
        )

        for part in self.discrete_labels:
            if self.argmax_discrete:
                encoded[part] = self.encoder[part](batch[part].argmax(1))
            else:
                encoded[part] = self.encoder[part](batch[part])

        if len(self.continuous_labels) > 1:
            encoded[self.continuous_labels[-1]] = (
                self.encoder[self.continuous_labels[0]](batch[self.comb_label])
                @ self.encoder[self.continuous_labels[-1]].weight
            )
        return encoded

    def latent_compose_function(self, z_parts, **kwargs):
        latent_basal = z_parts[self.hparams.x_label]
        if len(self.continuous_labels) > 1:
            latent_perturbation = z_parts[self.continuous_labels[-1]]
            latent_basal = latent_basal + latent_perturbation
        latent_covariate = 0
        for i in self.discrete_labels:
            latent_covariate += z_parts[i]
        latent_basal = latent_basal + latent_covariate
        return latent_basal

    def model_step(self, stage, batch, batch_idx):

        (
            x_hat,
            z_parts,
            z_parts_params,
            z_composed,
            loss,
            reconstruction_loss,
            kld_loss,
            kld_per_part,
        ) = self.forward(batch, decode=True, compute_loss=True)
        # if stage == 'train':
        #     print(reconstruction_loss, z_composed.max(), batch['drug_dose'].unique())

        mu = z_parts_params[self.hparams.x_label]
        if mu.shape[1] != self.latent_dim:
            mu = mu[:, : int(mu.shape[1] / 2)]

        _loss = {}

        weighted_adv_loss = 0
        adv_loss = 0
        for part in self.latent_loss.keys():
            if isinstance(self.latent_loss[part].loss, torch.nn.modules.loss.BCEWithLogitsLoss):
                batch[self.latent_loss_target[part]] = batch[self.latent_loss_target[part]].gt(0)

            _loss[part] = self.latent_loss[part](mu, batch[self.latent_loss_target[part]])
            adv_loss += _loss[part]
            weighted_adv_loss += _loss[part] * self.latent_loss_weights.get(part, 1.0)

        if stage != "test":
            optimizers = self.optimizers()
            lr_schedulers = self.lr_schedulers()

            main_optim = optimizers.pop(0)
            _ = lr_schedulers.pop(0)

            non_main_key = [i for i in self.optimizer.keys() if i != "main"]
            non_main_optims = []
            non_main_lr_scheds = []
            for i in non_main_key:
                non_main_optims.append(optimizers.pop(0))
                non_main_lr_scheds.append(lr_schedulers.pop(0))

            adversarial_flag = False
            for optim_ix, (optim, lr_sched) in enumerate(zip(optimizers, lr_schedulers)):

                group_key = self.latent_loss_optimizer_map[optim_ix]
                mod = self.latent_loss_backprop_when.get(group_key) or 3

                if stage == "train":
                    if batch_idx % mod == 0:
                        adversarial_flag = True
                        optim.zero_grad()
                        self.manual_backward(adv_loss)
                        optim.step()
                        # Dont use LR scheduler here, messes up the loss
                        if lr_sched is not None:
                            lr_sched.step()

            if (stage == "train") and (not adversarial_flag):
                main_optim.zero_grad()
                for non_main_optim in non_main_optims:
                    non_main_optim.zero_grad()

                self.manual_backward(
                    reconstruction_loss + self.beta * kld_loss - weighted_adv_loss
                )
                # self.manual_backward(reconstruction_loss - adv_loss)
                main_optim.step()
                for non_main_optim in non_main_optims:
                    non_main_optim.step()
                # Dont use LR scheduler here, messes up the loss
                # if main_lr_sched is not None:
                #     main_lr_sched.step()
                # for non_main_lr_sched in non_main_lr_scheds:
                #     non_main_lr_sched.step()

        loss = reconstruction_loss + self.beta * kld_loss - weighted_adv_loss

        results = self.make_results_dict(
            stage,
            batch,
            loss,
            reconstruction_loss,
            kld_loss,
            kld_per_part,
            z_parts,
            z_parts_params,
            z_composed,
            x_hat,
        )

        for part, value in _loss.items():
            results.update(
                {
                    f"adv_loss/{part}": value.detach().cpu(),
                }
            )

        # self.log_metrics(stage, results, x_hat.shape[0])

        return results, None, None

    def configure_optimizers(self):
        def get_params(obj):
            return list(obj.parameters())

        _parameters = get_params(self.decoder)
        for part in self.optimizer["main"]["keys"]:
            _parameters.extend(get_params(self.encoder[part]))

        optimizers = [self.optimizer["main"]["opt"](_parameters)]

        lr_schedulers = [
            (
                self.lr_scheduler["main"](optimizer=optimizers[0])
                if self.lr_scheduler["main"] is not None
                else None
            )
        ]

        non_main_key = [i for i in self.optimizer.keys() if i != "main"]

        if len(non_main_key) > 0:
            non_main_key = non_main_key[0]
            _parameters2 = get_params(self.encoder[self.optimizer[non_main_key]["keys"][0]])
            if len(self.optimizer[non_main_key]["keys"]) > 1:
                for key in self.optimizer[non_main_key]["keys"][1:]:
                    _parameters2.extend(
                        get_params(self.encoder[self.optimizer[non_main_key]["keys"][key]])
                    )
            optimizers.append(self.optimizer[non_main_key]["opt"](_parameters2))
            lr_schedulers.append(self.lr_scheduler[non_main_key](optimizer=optimizers[-1]))

        self.latent_loss_optimizer_map = {}
        for optim_ix, (group_key, group) in enumerate(self.latent_loss_optimizer.items()):
            self.latent_loss_optimizer_map[optim_ix] = group_key
            _parameters3 = get_params(self.latent_loss[group["keys"][0]])
            if len(group["keys"]) > 1:
                for key in group["keys"][1:]:
                    _parameters3.extend(get_params(self.latent_loss[key]))
            optimizers.append(group["opt"](_parameters3))
            lr_schedulers.append(self.latent_loss_scheduler[group_key](optimizer=optimizers[-1]))

        return optimizers, lr_schedulers
