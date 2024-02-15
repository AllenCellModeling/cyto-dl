import sys
from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.nn as nn
from monai.data.meta_tensor import MetaTensor
from monai.inferers import sliding_window_inference
from torchmetrics import MeanMetric

from cyto_dl.models.base_model import BaseModel


class MultiTaskIm2Im(BaseModel):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        task_heads: Dict,
        x_key: str,
        save_dir="./",
        save_images_every_n_epochs=1,
        inference_args: Dict = {},
        inference_heads: Union[List, None] = None,
        compile=False,
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
        inference_heads: Union[List, None] = None
            Optional list of heads to run during inference. Defaults to running all heads.
        compile: False
            Whether to compile the model using torch.compile
        **base_kwargs:
            Additional arguments passed to BaseModel
        """

        _DEFAULT_METRICS = {
            "train/loss": MeanMetric(),
            "val/loss": MeanMetric(),
            "test/loss": MeanMetric(),
        }

        for head in task_heads.keys():
            _DEFAULT_METRICS.update(
                {
                    f"train/loss/{head}": MeanMetric(),
                    f"val/loss/{head}": MeanMetric(),
                    f"test/loss/{head}": MeanMetric(),
                }
            )

        metrics = base_kwargs.pop("metrics", _DEFAULT_METRICS)
        super().__init__(metrics=metrics, **base_kwargs)

        self.automatic_optimization = True

        if compile is True and not sys.platform.startswith("win"):
            self.backbone = torch.compile(backbone)
            self.task_heads = torch.nn.ModuleDict(
                {k: torch.compile(v) for k, v in task_heads.items()}
            )
        else:
            self.backbone = backbone
            self.task_heads = torch.nn.ModuleDict(task_heads)

        self.inference_heads = inference_heads or list(self.task_heads.keys())

        for k, head in self.task_heads.items():
            head.update_params({"head_name": k, "x_key": x_key, "save_dir": save_dir})

    def configure_optimizers(self):
        opts = []
        scheds = []
        for key in ("generator", "discriminator"):
            if key in self.optimizer.keys():
                if key == "generator":
                    opt = self.optimizer[key](
                        filter(
                            lambda p: p.requires_grad,
                            list(self.backbone.parameters()) + list(self.task_heads.parameters()),
                        )
                    )
                elif key == "discriminator":
                    opt = self.optimizer[key](self.discriminator.parameters())
                scheduler = self.lr_scheduler[key](optimizer=opt)
                opts.append(opt)
                scheds.append(scheduler)
        return (opts, scheds)

    def _train_forward(self, batch, stage, save_image, run_heads):
        """during training we are only dealing with patches,so we can calculate per-patch loss,
        metrics, postprocessing etc."""
        z = self.backbone(batch[self.hparams.x_key])
        return {
            task: self.task_heads[task].run_head(z, batch, stage, save_image, self.global_step)
            for task in run_heads
        }

    def forward(self, x, run_heads):
        z = self.backbone(x)
        return {task: self.task_heads[task](z) for task in run_heads}

    def _inference_forward(self, batch, stage, save_image, run_heads):
        """during inference, we need to calculate per-fov loss/metrics/postprocessing.

        To avoid storing and passing to each head the intermediate results of the backbone, we need
        to run backbone + taskheads patch by patch, then do saving/postprocessing/etc on the entire
        fov.
        """
        with torch.no_grad():
            raw_pred_images = sliding_window_inference(
                inputs=batch[self.hparams.x_key],
                predictor=self.forward,
                run_heads=run_heads,
                **self.hparams.inference_args,
            )
        return {
            head_name: self.task_heads[head_name].run_head(
                None,
                batch,
                stage,
                save_image,
                self.global_step,
                run_forward=False,
                y_hat=raw_pred_images[head_name],
            )
            for head_name in run_heads
        }

    def run_forward(self, batch, stage, save_image, run_heads):
        if stage in ("train", "val"):
            return self._train_forward(batch, stage, save_image, run_heads)
        return self._inference_forward(batch, stage, save_image, run_heads)

    def should_save_image(self, batch_idx, stage):
        return stage in ("test", "predict") or (
            batch_idx < len(self.task_heads)  # noqa: FURB124
            and (self.current_epoch + 1) % self.hparams.save_images_every_n_epochs == 0
        )

    def _sum_losses(self, losses):
        summ = 0
        for k, v in losses.items():
            summ += v
        losses["loss"] = summ
        return losses

    def _get_run_heads(self, batch, stage):
        if stage not in ("test", "predict"):
            run_heads = [key for key in self.task_heads.keys() if key in batch]
        else:
            run_heads = self.inference_heads
        return run_heads

    def model_step(self, stage, batch, batch_idx):
        batch["filenames"] = batch[self.hparams.x_key].meta.get("filename_or_obj", batch_idx)
        # convert monai metatensors to tensors
        for k, v in batch.items():
            if isinstance(v, MetaTensor):
                batch[k] = v.as_tensor()

        run_heads = self._get_run_heads(batch, stage)
        outs = self.run_forward(batch, stage, self.should_save_image(batch_idx, stage), run_heads)

        losses = {head_name: head_result["loss"] for head_name, head_result in outs.items()}
        losses = self._sum_losses(losses)
        return losses, None, None

    def predict_step(self, batch, batch_idx):
        stage = "predict"
        batch["filenames"] = batch[self.hparams.x_key].meta["filename_or_obj"]
        # convert monai metatensors to tensors
        for k, v in batch.items():
            if isinstance(v, MetaTensor):
                batch[k] = v.as_tensor()
        run_heads = self._get_run_heads(batch, stage)
        outs = self.run_forward(batch, stage, self.should_save_image(batch_idx, stage), run_heads)
        # create input-> per head output mapping
        io_map = {}
        for head, output in outs.items():
            head_io_map = output["save_path"]
            for in_file, out_file in zip(head_io_map["input"], head_io_map["output"]):
                if in_file not in io_map:
                    io_map[in_file] = {}
                io_map[in_file][head] = out_file
        return io_map
