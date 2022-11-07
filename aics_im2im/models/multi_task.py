from typing import Optional, Union, Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss as Loss
from aicsimageio.writers import OmeTiffWriter
from pathlib import Path
import numpy as np

from serotiny.models.base_model import BaseModel
from serotiny.ml_ops.mlflow_utils import upload_artifacts
from monai.inferers import sliding_window_inference

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
        buffer_update_frequency: int = 50,
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

        self.task_heads = {}
        self.losses = {}
        for task, task_dict in tasks.items():
            self.task_heads[task] = task_dict.head
            self.losses[task] = task_dict.loss
        self.task_heads = torch.nn.ModuleDict(self.task_heads)
        
        self.automatic_optimization = automatic_optimization
        self.filenames = {}

    def on_train_start(self, **kwargs):
        # start background thread for loading images to update cache
        self.trainer.train_dataloader.dataset.datasets.start()

    def on_train_end(self, **kwargs):
        # shutdown background cache filler threads
        self.trainer.train_dataloader.dataset.datasets.shutdown()

    def forward(self, x):
        z = self.backbone(x)
        return {task: head(z) for task, head in self.task_heads.items()}

    def save_tensor(self, fn, img):
        img = img.detach().cpu().numpy()
        # img *= 255
        # img = np.clip(img, 0, 255)
        with upload_artifacts("images") as save_dir:
            OmeTiffWriter().save(
                uri=Path(save_dir) / fn,
                data=img,  # .astype(np.uint8),
                dims_order="STCZYX"[-len(img.shape)],
            )

    def _calculate_iou(self, im1, im2):
        return (np.sum(np.logical_and(im1, im2)) + 1e-8) / (
            np.sum(np.logical_or(im1, im2)) + 1e-8
        )

    def calculate_channelwise_iou(self, target, pred):
        iou_dict = {}
        for key in target.keys():
            for ch in range(target[key].shape[1]):
                iou_dict[f"{key}_{ch}"] = self._calculate_iou(
                    target[key][:, ch], pred[key][:, ch] > 0.5
                )
        return iou_dict

    def on_train_epoch_end(self, **kwargs):
        # update buffer by replacing percentage of images
        if (self.current_epoch + 1) % self.hparams.buffer_update_frequency == 0:
            self.trainer.train_dataloader.dataset.datasets.update_cache()
            # self.trainer.val_dataloaders[0].dataset.update_cache()

    def _step(self, stage, batch, batch_idx, logger):
        x = batch[self.hparams.x_key]
        targets = {k: batch[k] for k in self.task_heads.keys()}

        if stage != "test":
            outs = self(x)
            if (
                batch_idx == 0
                and self.current_epoch + 1 % self.hparams.save_images_every_n_epochs
            ):
                for k, v in batch.items():
                    self.save_tensor(f"{stage}_{k}_{self.global_step}.tif", v)
                for k, v in outs.items():
                    self.save_tensor(f"{stage}_{k}_{self.global_step}_pred.tif", v)
        else:
            metadata = batch[f"{self.hparams.x_key}_meta_dict"]
            source_filename = str(Path(metadata["filename_or_obj"][0]).stem) + "ome.tif"
            with torch.no_grad():
                outs = sliding_window_inference(
                    inputs=x,
                    roi_size=[32, 64, 64],
                    sw_batch_size=4,
                    predictor=self.forward,
                )
            # TODO go from dict to multichannel output here
            for k, v in outs.items():
                self.save_tensor(f"{k}_{source_filename}", v)

        if stage != "train":
            iou_dict = self.calculate_channelwise_iou(targets, outs)
            self.log_dict(
                {f"{stage}_{k}_iou": v for k, v in iou_dict.items()},
                logger=True,
                sync_dist=True,
            )

        losses = {
            f"{task}_loss": self.losses[task](task_out, targets[task].as_tensor())
            for task, task_out in outs.items()
        }

        summ = 0
        for k, v in losses.items():
            summ += v
        losses["loss"] = summ

        self.log_dict(
            {f"{stage}_{k}": v.detach() for k, v in losses.items()},
            logger=True,
            sync_dist=True,
        )
        if self.automatic_optimization or stage == "train":
            return losses
