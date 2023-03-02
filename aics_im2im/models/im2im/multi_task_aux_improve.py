from pathlib import Path
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from aicsimageio.writers import OmeTiffWriter
from monai.data.meta_tensor import MetaTensor
from monai.inferers import sliding_window_inference

from aics_im2im.models.base_model import BaseModel


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
        task_heads: Dict,
        x_key: str,
        save_dir="./",
        save_images_every_n_epochs=1,
        optimizer=torch.optim.Adam,
        automatic_optimization: bool = True,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        for stage in ("train", "val", "test", "predict"):
            (Path(save_dir) / f"{stage}_images").mkdir(exist_ok=True, parents=True)
        self.backbone = backbone
        self.task_heads = torch.nn.ModuleDict(task_heads)
        for k, head in self.task_heads.items():
            head.update_params({
                'head_name': k,
                'x_key': x_key,
                'save_dir': save_dir
            })

    def configure_optimizers(self):
        opts = []
        scheds = []
        for key in ("generator", "discriminator"):
            if key in self.hparams.optimizer.keys():
                if key == "generator":
                    opt = self.hparams.optimizer[key](list(self.backbone.parameters()) + list(self.task_heads.parameters()))
                elif key == "discriminator":
                    opt = self.hparams.optimizer[key](self.discriminator.parameters())
                scheduler = self.hparams.lr_scheduler[key](optimizer=opt)
                opts.append(opt)
                scheds.append(scheduler)
        return (opts, scheds)

    def forward(self, batch, stage, save_image):
        x= batch[self.hparams.x_key]
        z = self.backbone(x)
        return {task: head.run_head(z, batch, stage, save_image, self.global_step) for task, head in self.task_heads.items()}

    def should_save_image(self, batch_idx,stage):
        return stage in ('test', 'predict') or (
            batch_idx == 0  # noqa: FURB124
            and (self.current_epoch + 1) % self.hparams.save_images_every_n_epochs == 0
        )

    def _step(self, stage, batch, batch_idx, logger, optimizer_idx=0):
        # convert monai metatensors to tensors
        for k, v in batch.items():
            if isinstance(v, MetaTensor):
                batch[k] = v.as_tensor()

        outs = self(batch, stage, self.should_save_image(batch_idx, stage))        
        if stage == "predict":
            return        
        losses = {head_name: head_result['loss'] for head_name, head_result in outs.items()}
        losses = sum_losses(losses)
        self.log_dict(
            {f"{stage}_{k}": v for k, v in losses.items()},
            logger=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )
        return losses
