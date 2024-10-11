from cyto_dl.nn.head import BaseHead


class MAEHead(BaseHead):
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
            y_hat, mask = backbone_features
        else:
            raise ValueError("MAE head is only intended for use during training.")

        loss = (batch[self.head_name] - y_hat) ** 2
        if mask.sum() > 0:
            loss = loss[mask.bool()].mean()
        else:
            loss = loss.mean()

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
