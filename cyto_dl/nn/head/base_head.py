from abc import ABC
from pathlib import Path

import torch
from bioio.writers import OmeTiffWriter

from cyto_dl.models.im2im.utils.postprocessing import detach


class BaseHead(ABC, torch.nn.Module):
    """Base class for task heads."""

    def __init__(
        self,
        loss,
        postprocess={"input": detach, "prediction": detach},
        save_input=False,
    ):
        """
        Parameters
        ----------
        loss
            Loss function for task
        postprocess={"input": detach, "prediction": detach}
            Postprocessing for `input` and `predictions` of head
        save_input=False
            Whether to save out example input images during training
        """
        super().__init__()
        self.loss = loss
        self.postprocess = postprocess

        self.model = torch.nn.Sequential(torch.nn.Identity())
        self.save_input = save_input

    def update_params(self, params):
        for k, v in params.items():
            setattr(self, k, v)

    def _calculate_loss(self, y_hat, y):
        return self.loss(y_hat, y)

    def _postprocess(self, img, img_type):
        return [self.postprocess[img_type](img[i]) for i in range(img.shape[0])]

    def generate_io_map(self, meta, stage, batch_idx, step):
        """generates map between input files and output files for a head."""
        # filename is determined by step in training during train/val and by its source filename for prediction/testing
        filename_map = {"input": meta.get("filename_or_obj", [batch_idx])}
        if stage in ("train", "val", "test"):
            out_paths = [Path(self.save_dir) / f"{stage}_images" / f"{step}_{self.head_name}.tif"]
        else:
            out_paths = [
                Path(self.save_dir) / self.head_name / f"{Path(fn).stem}.tif"
                for fn in filename_map["input"]
            ]
        # create output directory if it doesn't exist
        out_paths[0].parent.mkdir(exist_ok=True, parents=True)

        filename_map["output"] = out_paths
        self.filename_map = filename_map
        return filename_map

    def save_image(self, y_hat, batch, stage):
        y_hat_out = self._postprocess(y_hat, img_type="prediction")
        y_out = None
        for i, out_path in enumerate(self.filename_map["output"]):
            OmeTiffWriter.save(data=y_hat_out[i], uri=out_path)
            if stage in ("train", "val"):
                y_out = self._postprocess(batch[self.head_name], img_type="input")
                OmeTiffWriter.save(data=y_out[i], uri=str(out_path).replace(".t", "_label.t"))
                if self.save_input:
                    raw_out = self._postprocess(batch[self.x_key][i : i + 1], img_type="input")
                    OmeTiffWriter.save(data=raw_out, uri=str(out_path).replace(".t", "_input.t"))
        return y_hat_out, y_out

    def forward(self, x):
        return self.model(x)

    def run_head(
        self,
        backbone_features,
        batch,
        stage,
        save_image,
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
            y_hat_out, y_out = self.save_image(y_hat, batch, stage)

        return {
            "loss": loss,
            "y_hat_out": y_hat_out,
            "y_out": y_out,
        }
