import sys
from typing import Dict

import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from torchmetrics import MeanMetric

from cyto_dl.models.im2im.multi_task import MultiTaskIm2Im

_DEFAULT_METRICS = {
    "train/loss/discriminator_loss": MeanMetric(),
    "val/loss/discriminator_loss": MeanMetric(),
    "test/loss/discriminator_loss": MeanMetric(),
    "train/loss/generator_loss": MeanMetric(),
    "val/loss/generator_loss": MeanMetric(),
    "test/loss/generator_loss": MeanMetric(),
    "train/loss": MeanMetric(),
    "val/loss": MeanMetric(),
    "test/loss": MeanMetric(),
}


class GAN(MultiTaskIm2Im):
    """Basic GAN model."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        task_heads: Dict[str, nn.Module],
        discriminator: nn.Module,
        x_key: str,
        save_dir="./",
        save_images_every_n_epochs=1,
        automatic_optimization: bool = False,
        inference_args: Dict = {},
        compile: False,
        **base_kwargs,
    ):
        """
        Parameters
        ----------
        backbone: nn.Module
            backbone network, parameters are shared between task heads
        task_heads: Dict
            task-specific heads
        discriminator
            discriminator network
        x_key: str
            key of input image in batch
        save_dir="./"
            directory to save images during training and validation
        save_images_every_n_epochs=1
            Frequency to save out images during training
        inference_args: Dict = {}
            Arguments passed to monai's [sliding window inferer](https://docs.monai.io/en/stable/inferers.html#sliding-window-inference)
        compile: False
            Whether to compile the model using torch.compile
        **base_kwargs:
            Additional arguments passed to BaseModel
        """

        metrics = base_kwargs.pop("metrics", _DEFAULT_METRICS)
        super().__init__(
            metrics=metrics, backbone=backbone, task_heads=task_heads, x_key=x_key, **base_kwargs
        )
        self.automatic_optimization = False

        if compile is True and not sys.platform.startswith("win"):
            self.discriminator = torch.compile(discriminator)
        else:
            self.discriminator = discriminator

        assert len(self.task_heads.keys()) == 1, "Only single-head GANs are supported currently."
        self.inference_heads = list(self.task_heads.keys())

        for k, head in self.task_heads.items():
            head.update_params({"head_name": k, "x_key": x_key, "save_dir": save_dir})

    def configure_optimizers(self):
        opts = []
        scheds = []
        for key in ("generator", "discriminator"):
            if key in self.optimizer.keys():
                if key == "generator":
                    opt = self.optimizer[key](
                        list(self.backbone.parameters()) + list(self.task_heads.parameters())
                    )
                elif key == "discriminator":
                    opt = self.optimizer[key](self.discriminator.parameters())
                scheduler = self.lr_scheduler[key](optimizer=opt)
                opts.append(opt)
                scheds.append(scheduler)
        return (opts, scheds)

    def _train_forward(self, batch, stage, save_image, run_heads):
        """During training we are only dealing with patches,so we can calculate per-patch loss,
        metrics, postprocessing etc."""
        z = self.backbone(batch[self.hparams.x_key])
        return {
            task: self.task_heads[task].run_head(
                z, batch, stage, save_image, discriminator=self.discriminator
            )
            for task in run_heads
        }

    def _inference_forward(self, batch, stage, save_image, run_heads):
        """During inference, we need to calculate per-fov loss/metrics/postprocessing.

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
            head_name: head.run_head(
                None,
                batch,
                stage,
                save_image,
                discriminator=self.discriminator if stage == "test" else None,
                run_forward=False,
                y_hat=raw_pred_images[head_name],
            )
            for head_name, head in self.task_heads.items()
        }

    def _extract_loss(self, outs, loss_type):
        loss = {
            f"{head_name}_{loss_type}": head_result[loss_type]
            for head_name, head_result in outs.items()
        }
        return self._sum_losses(loss)

    def model_step(self, stage, batch, batch_idx):
        run_heads, _ = self._get_run_heads(batch, stage, batch_idx)
        n_postprocess = self.get_n_postprocess_image(batch, batch_idx, stage)

        batch = self._to_tensor(batch)
        outs = self.run_forward(batch, stage, n_postprocess, run_heads)

        loss_D = self._extract_loss(outs, "loss_D")
        loss_G = self._extract_loss(outs, "loss_G")

        if stage == "train":
            g_opt, d_opt = self.optimizers()

            g_opt.zero_grad()
            self.manual_backward(loss_G["loss"])
            g_opt.step()

            d_opt.zero_grad()
            self.manual_backward(loss_D["loss"])
            d_opt.step()
        results = {f"discriminator_{key}": loss for key, loss in loss_D.items()}
        results.update({f"generator_{key}": loss for key, loss in loss_G.items()})
        results["loss"] = results["generator_loss"]

        if n_postprocess > 0:
            # add postprocessed images to return dict
            for k in ("pred", "target", "input"):
                results[k] = self.get_per_head(outs, k)

        self.compute_metrics(results, None, None, stage)
        return results

    def predict_step(self, batch, batch_idx):
        stage = "predict"
        run_heads, io_map = self._get_run_heads(batch, stage, batch_idx)
        outs = None
        if len(run_heads) > 0:
            n_postprocess = self.get_n_postprocess_image(batch, batch_idx, stage)
            batch = self._to_tensor(batch)
            outs = self.run_forward(batch, stage, n_postprocess, run_heads)
        return io_map, outs
