from typing import Optional, Union, Callable, Dict

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss as Loss

from serotiny.models import BaseModel

class MultiTaskIm2Im(BaseModel):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        tasks: Dict,
        x_key: str,
        reduce_task_outs: str = "concat",
        automatic_optimization: bool = False,
        **kwargs
    ):
        super().__init__(
            backbone=backbone,
            tasks=tasks,
            x_key=x_key,
            reduce_task_outs=reduce_task_outs,
            automatic_optimization=automatic_optimization,
            **kwargs
        )

        self.backbone = backbone

        self.task_heads = {}
        self.losses = {}
        for task, task_dict in tasks.items():
            self.task_heads[task] = task_dict.head
            self.losses[task] = task_dict.loss


        if reduce_task_outs not in ("concat", "sum", "mean"):
            raise ValueError

        self.reduce_task_outs = reduce_task_outs
        self.automatic_optimization = automatic_optimization

    def forward(self, x):
        z = self.backbone(x)
        return {task: head(z) for task, head in self.task_heads.items()}

        return {"common": self.common_head(reduced_task_outs), **task_outs}

    def _step(self, stage, batch, batch_idx, logger):
        x = batch[self.hparams.x_key]
        targets = {k: batch[k] for k in self.task_heads.keys()}

        outs = self(x)

        # TODO: implement step logic here. use variable `stage`
        # to distinguish between train/val/test

        losses = {self.losses[task](task_out, targets[task])
                  for task, task_out in outs.items()}

        loss = self.compute_loss(x, targets)

        if self.automatic_optimization or stage == "train":
            return loss

    def configure_optimizers(self):
        raise NotImplementedError


    def reduce_task_outs(self, task_outs):
        if self.reduce_task_outs == "concat":
            return torch.concat(
                task_outs[sorted(task_outs.keys())].values(),
                axis=1
            )
        elif self.reduce_task_outs == "sum":
            return torch.sum(
                task_outs.values(),
                axis=1
            )
        return torch.mean(
            task_outs.values(),
            axis=1
        )
