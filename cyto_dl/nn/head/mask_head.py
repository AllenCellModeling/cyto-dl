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
        """
        super().__init__(loss, postprocess=postprocess)
        self.mask_key = mask_key

        self.model = torch.nn.Sequential(torch.nn.Identity())

    def _calculate_loss(self, y_hat, y, mask):
        return self.loss(y_hat, y, mask)

    def run_head(
        self,
        backbone_features,
        batch,
        stage,
        n_postprocess,
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
