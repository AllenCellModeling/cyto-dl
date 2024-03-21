from abc import ABC
from pathlib import Path

import torch
from aicsimageio.writers import OmeTiffWriter

from cyto_dl.models.im2im.utils.postprocessing import detach


class PointCloudHead(ABC, torch.nn.Module):
    """Base class for task heads."""

    def __init__(
        self,
        loss,
        postprocess={"input": detach, "prediction": detach},
        calculate_metric=False,
        save_raw=False,
        scale_loss: float = 0.0001,
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
        self.scale_loss = scale_loss

    def update_params(self, params):
        for k, v in params.items():
            setattr(self, k, v)

    def _calculate_loss(self, y_hat, y):
        return self.loss(y_hat, y)

    def _postprocess(self, img, img_type):
        return [self.postprocess[img_type](img[i]) for i in range(img.shape[0])]

    def _save(self, fn, out, stage):
        out_path = Path(self.save_dir) / f"{stage}_pointclouds" / fn

        out = pd.DataFrame(out.detach().cpu().numpy(), columns=["z", "y", "x", "s"])
        cloud = PyntCloud(out_path)
        cloud.to_file(out_path)
        return out_path

    def _calculate_metric(self, y_hat, y):
        raise NotImplementedError

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

        y_hat = self._postprocess(y_hat, img_type="prediction")

        if isinstance(batch[self.head_name], list):
            batch[self.head_name] = torch.stack(batch[self.head_name], dim=0)

        loss = None
        if stage != "predict":
            loss = self._calculate_loss(y_hat, batch[self.head_name])

        loss = loss.mean() * self.scale_loss

        return {
            "loss": loss,
            "metric": None,
            "y_hat_out": y_hat,
            "y_out": batch[self.head_name],
            "save_path": None,
        }
