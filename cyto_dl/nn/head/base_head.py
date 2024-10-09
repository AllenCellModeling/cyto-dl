from abc import ABC
from pathlib import Path

import torch

from cyto_dl.models.im2im.utils.postprocessing import detach


class BaseHead(ABC, torch.nn.Module):
    """Base class for task heads."""

    def __init__(
        self,
        loss,
        postprocess={"input": detach, "prediction": detach},
    ):
        """
        Parameters
        ----------
        loss
            Loss function for task
        postprocess={"input": detach, "prediction": detach}
            Postprocessing for `input` and `predictions` of head
        """
        super().__init__()
        self.loss = loss
        self.postprocess = postprocess

        self.model = torch.nn.Sequential(torch.nn.Identity())

    def update_params(self, params):
        for k, v in params.items():
            setattr(self, k, v)

    def _calculate_loss(self, y_hat, y):
        return self.loss(y_hat, y)

    def _postprocess(self, img, img_type, n_postprocess=1):
        return [self.postprocess[img_type](img[i]) for i in range(n_postprocess)]

    def generate_io_map(self, input_filenames):
        """Generates map between input files and output files for a head.

        Only used for prediction
        """
        filename_map = {"input": input_filenames}
        filename_map["output"] = [
            Path(self.save_dir) / self.head_name / f"{Path(fn).stem}.tif"
            for fn in filename_map["input"]
        ]
        # create output directory if it doesn't exist
        filename_map["output"][0].parent.mkdir(exist_ok=True, parents=True)
        return filename_map

    def forward(self, x):
        return self.model(x)

    def run_head(
        self,
        backbone_features,
        batch,
        stage,
        n_postprocess=1,
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

        # no need to postprocess input and target during prediction
        return {
            "loss": loss,
            "pred": self._postprocess(y_hat, img_type="prediction", n_postprocess=n_postprocess),
            "target": (
                self._postprocess(
                    batch[self.head_name], img_type="input", n_postprocess=n_postprocess
                )
                if stage != "predict"
                else None
            ),
            "input": (
                self._postprocess(batch[self.x_key], img_type="input", n_postprocess=n_postprocess)
                if stage != "predict"
                else None
            ),
        }
