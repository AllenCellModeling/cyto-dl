from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from monai.data.meta_tensor import MetaTensor

from aics_im2im.models.base_model import BaseModel


class MultiTaskIm2Im(BaseModel):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        task_heads: Dict,
        x_key: str,
        save_dir="./",
        head_allocation_column: str = None,
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
            head.update_params({"head_name": k, "x_key": x_key, "save_dir": save_dir})

    def configure_optimizers(self):
        opts = []
        scheds = []
        for key in ("generator", "discriminator"):
            if key in self.hparams.optimizer.keys():
                if key == "generator":
                    opt = self.hparams.optimizer[key](
                        list(self.backbone.parameters()) + list(self.task_heads.parameters())
                    )
                elif key == "discriminator":
                    opt = self.hparams.optimizer[key](self.discriminator.parameters())
                scheduler = self.hparams.lr_scheduler[key](optimizer=opt)
                opts.append(opt)
                scheds.append(scheduler)
        return (opts, scheds)

    def forward(self, batch, stage, save_image, run_heads):
        # run all heads if head_allocation_column not in batch
        run_heads = run_heads or self.task_heads.keys()
        x = batch[self.hparams.x_key]
        z = self.backbone(x)
        return {
            task: self.task_heads[task].run_head(z, batch, stage, save_image, self.global_step)
            for task in run_heads
        }

    def should_save_image(self, batch_idx, stage):
        return stage in ("test", "predict") or (
            batch_idx == 0  # noqa: FURB124
            and (self.current_epoch + 1) % self.hparams.save_images_every_n_epochs == 0
        )

    def _sum_losses(self, losses):
        summ = 0
        for k, v in losses.items():
            summ += v
        losses["loss"] = summ
        return losses

    def _step(self, stage, batch, batch_idx, logger, optimizer_idx=0):
        # convert monai metatensors to tensors
        for k, v in batch.items():
            if isinstance(v, MetaTensor):
                batch[k] = v.as_tensor()

        run_heads = batch.get(self.hparams.head_allocation_column)

        outs = self(batch, stage, self.should_save_image(batch_idx, stage), run_heads)
        if stage == "predict":
            return
        losses = {head_name: head_result["loss"] for head_name, head_result in outs.items()}
        losses = self._sum_losses(losses)
        self.log_dict(
            {f"{stage}_{k}": v for k, v in losses.items()},
            logger=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )
        return losses
