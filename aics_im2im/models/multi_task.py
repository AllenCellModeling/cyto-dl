from typing import Optional, Union, Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss as Loss
from aicsimageio.writers import OmeTiffWriter
from pathlib import Path
import numpy as np
from monai.data.meta_tensor import MetaTensor

from serotiny.models.base_model import BaseModel
from serotiny.ml_ops.mlflow_utils import upload_artifacts
from monai.inferers import sliding_window_inference
from skimage.exposure import rescale_intensity


class MultiTaskIm2Im(BaseModel):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        tasks: Dict,
        x_key: str,
        save_images_every_n_epochs=1,
        optimizer=torch.optim.Adam,
        automatic_optimization: bool = True,
        patch_shape=[32, 128, 128],
        inference_heads=None,
        hr_skip=nn.Identity(),
        postprocessing=None,
        discriminator=None,
        gan_loss = lambda x: 0
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            tasks=tasks,
            x_key=x_key,
            automatic_optimization=automatic_optimization,
            **kwargs,
        )
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
        with upload_artifacts(directory) as save_dir:
            OmeTiffWriter().save(
                uri=Path(save_dir) / fn,
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

    def optimize_discriminator(self, targets, outs, stage):
        if stage != 'train' or self.discriminator is None:
            return
        assert not self.automatic_optimization, "automatic optimization must be off for GAN Training"
        d_opt=self.optimizers['D']
        self.discriminator.set_requires_grad(True)
        loss_D = 0
        for target_dict, target_type, d_target in zip(
            [targets, outs], ["real", "fake"], [True, False]
        ):
            disc_out = self.discriminator(target_dict)
            loss_D_partial = self.gan_loss(
                disc_out, d_target, iter=self.global_step
            )
            self.log(f"D_{target_type}", loss_D_partial, progress_bar=False)
            loss_D += loss_D_partial
        loss_D *= 0.5
        self.mnual_backward(loss_D)
        d_opt.step()
        self.discriminator.set_requires_grad(False)
        self.log('loss_D', loss_D)

    def optimize_generator(self, targets, outs, stage):
        if stage == 'train' and not self.automatic_optimization:
            g_opt = self.optimizers['generator']
            g_opt.zero_grad()
         losses = {
            f"{task}_loss": self.losses[task](task_out, targets[task])
            for task, task_out in outs.items()
        }
        pred_fake = self.discriminator(target_dict, detach=False)
        losses['g_gan'] = self.gan_loss(pred_fake, True)

        summ = 0
        for k, v in losses.items():
            summ += v
        losses["loss"] = summ

        self.log_dict(
            {f"{stage}_{k}": v for k, v in losses.items()},
            logger=True,
            sync_dist=True,
        )
        if stage =='train' and not self.automatic_optimization:
            self.manual_backward(losses['loss'])
            g_opt.step()
        
        return losses

    def should_save_image(self, batch_idx):
        return batch_idx == 0
                and (self.current_epoch + 1) % self.hparams.save_images_every_n_epochs
                == 0

    def _step(self, stage, batch, batch_idx, logger):
        # only need filename
        metadata = batch[f"{self.hparams.x_key}_meta_dict"]
        # convert monai metatensors to tensors
        for k, v in batch.items():
            if isinstance(v, MetaTensor):
                batch[k] = v.as_tensor()

        x = batch[self.hparams.x_key]
        targets = {k: batch[k] for k in self.task_heads.keys()}

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
            with torch.no_grad():
                outs = sliding_window_inference(
                    inputs=x,
                    roi_size=self.hparams.patch_shape,
                    sw_batch_size=4,
                    predictor=self.forward,
                    overlap=0.1,
                    mode="gaussian",
                    test=True,
                )
            # go from dict to multichannel output here
            for k, v in outs.items():
                for im_idx in range(v.shape[0]):
                    source_filename = (
                        str(Path(metadata["filename_or_obj"][im_idx]).stem) + ".ome.tif"
                    )
                    self.save_image(
                        f"{k}_{source_filename}", v[im_idx], directory=f"{stage}_images"
                    )

            if stage == "predict":
                return
        if stage in ["val", "test"]:
            iou_dict = self.calculate_channelwise_iou(targets, outs)
            self.log_dict(
                {f"{stage}_{k}_iou": v for k, v in iou_dict.items()},
                logger=True,
                sync_dist=True,
            )
        
        self.optimize_discriminator(targets, outs, stage)
        losses = self.optimize_generator(targets, outs, stage)

        if self.automatic_optimization or stage == "train":
            return losses
