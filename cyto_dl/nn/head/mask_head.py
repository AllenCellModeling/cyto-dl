import torch

from cyto_dl.models.im2im.utils.postprocessing import detach
from cyto_dl.nn.head import BaseHead


class MaskHead(BaseHead):
    """Task Head using a masked loss function."""

    def __init__(
        self,
        loss,
        mask_key: str = "mask",
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
        self.mask_key = mask_key

        self.model = torch.nn.Sequential(torch.nn.Identity())
        self.save_raw = save_raw

    def _calculate_loss(self, y_hat, y, mask):
        return self.loss(y_hat, y, mask)

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
            loss = self._calculate_loss(y_hat, batch[self.head_name], batch[self.mask_key])

        y_hat_out, y_out, out_paths = None, None, None
        if save_image:
            y_hat_out, y_out, out_paths = self.save_image(y_hat, batch, stage, global_step)

        metric = None
        if self.calculate_metric and stage in ("val", "test"):
            metric = self._calculate_metric(y_hat, batch[self.head_name])
        return {
            "loss": loss,
            "metric": metric,
            "y_hat_out": y_hat_out,
            "y_out": y_out,
            "save_path": out_paths,
        }
