from typing import List

from cyto_dl.nn import MLP
from cyto_dl.nn.head import BaseHead


class VICRegHead(BaseHead):
    def __init__(
        self, loss, dims: List[int] = [2048, 8192], hidden_layers: List[int] = [8192, 8192]
    ):
        """
        Parameters
        ----------
        loss
            Loss function for task
        dims
            input and output dimensions for Projectornetwork
        hidden_layers
            hidden layers for MLP
        """
        super().__init__(loss)
        self.model = MLP(*dims, hidden_layers=hidden_layers)

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
        x1, x2 = backbone_features
        x1, x2 = self.forward(x1), self.forward(x2)
        loss = self._calculate_loss(x1, x2)

        return {
            "loss": loss,
            "y_hat_out": x1,
            "y_out": x2,
        }
