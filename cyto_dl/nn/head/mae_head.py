from cyto_dl.nn.head import BaseHead


class MAEHead(BaseHead):
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
            y_hat, mask = backbone_features
        else:
            raise ValueError("MAE head is only intended for use during training.")
        loss = (batch[self.head_name] - y_hat) ** 2
        if mask.sum() > 0:
            loss = loss[mask.bool()].mean()
        else:
            loss = loss.mean()

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
