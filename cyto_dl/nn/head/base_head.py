import math
from abc import ABC
from pathlib import Path

import numpy as np
import torch
from aicsimageio.writers import OmeTiffWriter
from monai.networks.blocks import Convolution, UnetOutBlock, UnetResBlock, UpSample

from cyto_dl.models.im2im.utils.postprocessing import detach


class BaseHead(ABC, torch.nn.Module):
    """Base class for task heads."""

    def __init__(
        self,
        loss,
        postprocess={"input": detach, "prediction": detach},
        calculate_metric=False,
        save_raw=False,
    ):
        """
        Parameters
        ----------
        loss
            Loss function for task
        postprocess={"input": detach, "prediction": detach}
            Postprocessing for `input` and `predictions` of head
        calculate_metric=False
            Whether to calculate a metric during training. Not used by GAN head.
        save_raw=False
            Whether to save out example input images during training
        """
        super().__init__()
        self.loss = loss
        self.postprocess = postprocess
        self.calculate_metric = calculate_metric

        self.model = torch.nn.Sequential(torch.nn.Identity())
        self.save_raw = save_raw

    def update_params(self, params):
        for k, v in params.items():
            setattr(self, k, v)

    def _calculate_loss(self, y_hat, y):
        return self.loss(y_hat, y)

    def _postprocess(self, img, img_type):
        return [self.postprocess[img_type](img[i]) for i in range(img.shape[0])]

    def _save(self, fn, img, stage):
        OmeTiffWriter().save(
            uri=Path(self.save_dir) / f"{stage}_images" / fn,
            data=img.squeeze(),
            dims_order="STCZYX"[-len(img.shape)],
        )

    def _calculate_metric(self, y_hat, y):
        raise NotImplementedError

    def save_image(self, y_hat, batch, stage, global_step):
        y_hat_out = self._postprocess(y_hat, img_type="prediction")
        y_out, raw_out = None, None
        # filename is determined by step in training during train/val and by its source filename for prediction/testing
        if stage in ("train", "val"):
            y_out = self._postprocess(batch[self.head_name], img_type="input")
            if self.save_raw:
                raw_out = self._postprocess(batch[self.x_key], img_type="input")
        try:
            metadata_filenames = batch[f"{self.x_key}_meta_dict"]["filename_or_obj"]
            metadata_filenames = [
                f"{Path(fn).stem}_{self.head_name}.tif" for fn in metadata_filenames
            ]
        except KeyError:
            raise ValueError(
                f"Please ensure your batches contain key `{self.x_key}_meta_dict['filename_or_obj']`"
            )
        save_name = (
            [f"{global_step}_{self.head_name}.tif"]
            if stage in ("train", "val")
            else metadata_filenames
        )
        n_save = len(y_hat_out) if stage in ("test", "predict") else 1
        for i in range(n_save):
            self._save(save_name[i].replace(".tif", "_pred.tif"), y_hat_out[i], stage)
            if stage in ("train", "val"):
                self._save(save_name[i], y_out[i], stage)
                if self.save_raw:
                    self._save(save_name[i].replace(".tif", "_raw.tif"), raw_out[i], stage)

        return y_hat_out, y_out

    def forward(self, x):
        return self.model(x)

    def run_head(
        self,
        backbone_features,
        batch,
        stage,
        save_image,
        global_step,
        run_forward=True,
        y_hat=None,
    ):
        """Run head on backbone features, calculate loss, postprocess and save image, and calculate
        metrics."""
        if run_forward:
            y_hat = self.forward(backbone_features)
        if y_hat is None:
            raise ValueError(
                "y_hat must be provided, either by passing it in or setting `run_forward=True`"
            )
        loss = None
        if stage != "predict":
            loss = self._calculate_loss(y_hat, batch[self.head_name])

        y_hat_out, y_out = None, None
        if save_image:
            y_hat_out, y_out = self.save_image(y_hat, batch, stage, global_step)

        metric = None
        if self.calculate_metric and stage in ("val", "test"):
            metric = self._calculate_metric(y_hat, batch[self.head_name])
        return {"loss": loss, "metric": metric, "y_hat_out": y_hat_out, "y_out": y_out}
