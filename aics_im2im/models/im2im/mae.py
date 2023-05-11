from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from monai.data.meta_tensor import MetaTensor
from monai.inferers import sliding_window_inference
from torchmetrics import MeanMetric, MinMetric

from aics_im2im.models.base_model import BaseModel
from aicsimageio.writers import OmeTiffWriter


class MaskedAutoEncoder(BaseModel):
    def __init__(
        self,
        *,
        model,
        x_key,
        save_dir="./",
        save_images_every_n_epochs=1,
        inference_args: Dict = {},
        **base_kwargs,
    ):
        """
        Parameters
        ----------
        backbone: nn.Module
            backbone network, parameters are shared between task heads
        task_heads: Dict
            task-specific heads
        x_key: str
            key of input image in batch
        save_dir="./"
            directory to save images during training and validation
        save_images_every_n_epochs=1
            Frequency to save out images during training
        inference_args: Dict = {}
            Arguments passed to monai's [sliding window inferer](https://docs.monai.io/en/stable/inferers.html#sliding-window-inference)
        **base_kwargs:
            Additional arguments passed to BaseModel
        """

        _DEFAULT_METRICS = {
            "train/loss": MeanMetric(),
            "val/loss": MeanMetric(),
            "test/loss": MeanMetric(),
        }

        metrics = base_kwargs.pop("metrics", _DEFAULT_METRICS)
        super().__init__(metrics=metrics, **base_kwargs)

        self.automatic_optimization = True
        for stage in ("train", "val", "test", "predict"):
            (Path(save_dir) / f"{stage}_images").mkdir(exist_ok=True, parents=True)
        self.model = model #torch.compile(model) #model #

    def configure_optimizers(self):
        opts = []
        scheds = []
        for key in ("generator", "discriminator"):
            if key in self.optimizer.keys():
                if key == "generator":
                    opt = self.optimizer[key](self.model.parameters())
                scheduler = self.lr_scheduler[key](optimizer=opt)
                opts.append(opt)
                scheds.append(scheduler)
        return (opts, scheds)

    def forward(self, x):
        return self.model(x)

    def should_save_image(self, batch_idx, stage):
        return stage in ("test", "predict") or (
            batch_idx ==0   # noqa: FURB124
            and (self.current_epoch + 1) % self.hparams.save_images_every_n_epochs == 0
        )

    def model_step(self, stage, batch, batch_idx):
        # convert monai metatensors to tensors
        for k, v in batch.items():
            if isinstance(v, MetaTensor):
                batch[k] = v.as_tensor()

        x = batch[self.hparams.x_key]
        pred, mask = self(x)
        x = x[:, :, :pred.shape[-3], :pred.shape[-2], :pred.shape[-1]]

        loss = torch.mul((x-pred)**2, mask).mean()

        if self.should_save_image(batch_idx, stage):
            OmeTiffWriter().save(
                uri=Path(self.hparams.save_dir) / f"{stage}_images" / f'{self.current_epoch}_pred.tif',
                data=pred.detach().cpu().numpy().squeeze().astype(float),
                dims_order="STCZYX"[-len(pred.shape)],
            )
            OmeTiffWriter().save(
                uri=Path(self.hparams.save_dir) / f"{stage}_images" / f'{self.current_epoch}_real.tif',
                data=x.detach().cpu().numpy().squeeze().astype(float),
                dims_order="STCZYX"[-len(pred.shape)],
            )
            OmeTiffWriter().save(
                uri=Path(self.hparams.save_dir) / f"{stage}_images" / f'{self.current_epoch}_mask.tif',
                data=mask.detach().cpu().numpy().squeeze().astype(float),
                dims_order="STCZYX"[-len(pred.shape)],
            )

        return loss, pred, x
