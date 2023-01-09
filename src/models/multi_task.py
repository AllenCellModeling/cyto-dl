from typing import Optional, Union, Callable, Dict
import torch
import torch.nn as nn
from aicsimageio.writers import OmeTiffWriter
from pathlib import Path
import numpy as np
from monai.data.meta_tensor import MetaTensor

from serotiny.models.base_model import BaseModel

from monai.inferers import sliding_window_inference


def sum_losses(losses):
    summ = 0
    for k, v in losses.items():
        summ += v
    losses["loss"] = summ
    return losses


class MultiTaskIm2Im(BaseModel):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        tasks: Dict,
        x_key: str,
        save_dir="./",
        save_images_every_n_epochs=1,
        optimizer=torch.optim.Adam,
        automatic_optimization: bool = True,
        patch_shape=[32, 128, 128],
        inference_heads=None,
        hr_skip=nn.Identity(),
        postprocessing=None,
        discriminator=None,
        gan_loss=lambda x: 0,
        costmap_key="cmap",
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        for stage in ["train", "val", "test"]:
            (Path(save_dir) / f"{stage}_images").mkdir(exist_ok=True, parents=True)
        self.backbone = backbone
        self.hr_skip = hr_skip
        self.task_heads = {}
        self.losses = {}
        self.inference_heads = (
            tasks.keys() if inference_heads is None else inference_heads
        )
        for task, task_dict in tasks.items():
            self.task_heads[task] = task_dict.head
            self.losses[task] = task_dict.loss
        self.task_heads = torch.nn.ModuleDict(self.task_heads)
        self.automatic_optimization = automatic_optimization
        self.filenames = {}
        self.postprocessing = {} if postprocessing is None else postprocessing
        self.discriminator = discriminator
        self.gan_loss = gan_loss

    def configure_optimizers(self):
        opts = []
        scheds = []
        for key in ["generator", "discriminator"]:
            if key in self.hparams.optimizer.keys():
                if key == "generator":
                    opt = self.hparams.optimizer[key](
                        list(self.backbone.parameters())
                        + list(self.hr_skip.parameters())
                        + list(self.task_heads.parameters())
                    )
                elif key == "discriminator":
                    opt = self.hparams.optimizer[key](self.discriminator.parameters())
                scheduler = self.hparams.lr_scheduler[key](optimizer=opt)
                opts.append(opt)
                scheds.append(scheduler)
        return (opts, scheds)

    def forward(self, x, test=False):
        run_heads = self.task_heads.keys()
        if test:
            run_heads = self.inference_heads
        z = self.backbone(x)
        hr_skip = self.hr_skip(x)
        return {
            task: head(z, hr_skip)
            for task, head in self.task_heads.items()
            if task in run_heads
        }

    def save_image(self, fn, img, directory):
        OmeTiffWriter().save(
            uri=Path(self.hparams.save_dir) / directory / fn,
            data=img.squeeze(),
            dims_order="STCZYX"[-len(img.shape)],
        )

    def _calculate_iou(self, target, pred):
        target = target.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()

        # only calculate iou on binary targets
        if len(np.unique(target)) <= 2:
            return (np.sum(np.logical_and(target, pred)) + 1e-8) / (
                np.sum(np.logical_or(target, pred)) + 1e-8
            )
        else:
            return np.nan

    def calculate_channelwise_iou(self, target, pred):
        iou_dict = {}
        for key in target.keys():
            if not isinstance(target[key], torch.Tensor):
                continue
            for ch in range(target[key].shape[1]):
                iou_dict[f"{key}_{ch}"] = self._calculate_iou(
                    target[key][:, ch], pred[key][:, ch] > 0.5
                )
        return iou_dict

    def optimize_discriminator(self, targets, outs):
        if self.discriminator is None:
            return
        self.discriminator.set_requires_grad(True)
        loss_D = 0
        x = targets[self.hparams.x_key]
        for target_dict, target_type, d_target in zip(
            [targets, outs], ["real", "fake"], [True, False]
        ):
            disc_out = self.discriminator(target_dict, x)
            loss_D_partial = self.gan_loss(disc_out, d_target)
            self.log(f"D_{target_type}", loss_D_partial)
            loss_D += loss_D_partial
        loss_D *= 0.5
        self.discriminator.set_requires_grad(False)
        return {"loss_D": loss_D}

    def optimize_generator(self, targets, outs):
        losses = {
            f"{task}_loss": self.losses[task](task_out, targets[task])
            if self.hparams.costmap_key not in targets.keys()
            else self.losses[task](
                task_out, targets[task], targets[self.hparams.costmap_key]
            )
            for task, task_out in outs.items()
        }
        if self.discriminator is not None:
            pred_fake = self.discriminator(
                targets, targets[self.hparams.x_key], detach=False
            )
            losses["g_gan"] = self.gan_loss(pred_fake, True)
        return losses

    def should_save_image(self, batch_idx):
        return (
            batch_idx == 0
            and (self.current_epoch + 1) % self.hparams.save_images_every_n_epochs == 0
        )

    def _step(self, stage, batch, batch_idx, logger, optimizer_idx=0):
        # only need filename
        metadata = batch.get(f"{self.hparams.x_key}_meta_dict")
        # convert monai metatensors to tensors
        for k, v in batch.items():
            if isinstance(v, MetaTensor):
                batch[k] = v.as_tensor()

        x = batch[self.hparams.x_key]
        if stage in ["train", "val"]:
            outs = self(x)
            if self.should_save_image(batch_idx):
                for result, suffix, dict_type in zip(
                    [batch, outs], [".tif", "_pred.tif"], ["input", "prediction"]
                ):
                    for key in self.postprocessing[dict_type]:
                        if key not in result:
                            continue

                        self.save_image(
                            f"{self.global_step}_{key}{suffix}",
                            self.postprocessing[dict_type][key](result[key]),
                            directory=f"{stage}_images",
                        )

        elif stage in ["predict", "test"]:
            assert (
                metadata is not None
            ), "Metadata required for proper file saving during prediction! Please check your transforms"
            with torch.no_grad():
                outs = sliding_window_inference(
                    inputs=x,
                    roi_size=self.hparams.patch_shape,
                    sw_batch_size=16,
                    predictor=self.forward,
                    overlap=0.5,
                    mode="gaussian",
                    test=True,
                )
            for k, v in outs.items():
                for im_idx in range(v.shape[0]):
                    source_filename = (
                        str(Path(metadata["filename_or_obj"][im_idx]).stem) + ".ome.tif"
                    )
                    if k in self.postprocessing["prediction"]:
                        self.save_image(
                            f"{k}_{source_filename}",
                            self.postprocessing["prediction"][k](v[im_idx]),
                            directory=f"{stage}_images",
                        )

            if stage == "predict":
                return
        targets = {k: batch[k] for k in self.task_heads.keys()}
        if stage in ["val", "test"]:
            iou_dict = self.calculate_channelwise_iou(targets, outs)
            self.log_dict(
                {f"{stage}_{k}_iou": v for k, v in iou_dict.items()},
                logger=logger,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

        if optimizer_idx == 1:
            losses = self.optimize_discriminator(batch, outs)
        elif optimizer_idx in (0, None):
            losses = self.optimize_generator(batch, outs)

        losses = sum_losses(losses)

        self.log_dict(
            {f"{stage}_{k}": v for k, v in losses.items()},
            logger=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

        return losses