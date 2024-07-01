from torch import nn

from cyto_dl.nn.head import BaseHead


class Projector(nn.Module):
    def __init__(self, dimensions=[2048, 8192, 8192, 8192]):
        """
        Parameters
        ----------
        dimensions
            List of dimensions for projector network. Should start with output dimension of backone network. Subsequent dimensions should be ~4x larger following vicreg
        """
        super().__init__()
        layers = []
        for i in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            layers.append(nn.BatchNorm1d(dimensions[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(dimensions[-2], dimensions[-1], bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class VICRegHead(BaseHead):
    def __init__(
        self,
        loss,
        dimensions=[2048, 8192, 8192, 8192],
    ):
        """
        Parameters
        ----------
        loss
            Loss function for task
        dimensions
            List of dimensions for projector network. Should start with output dimension of backone network. Subsequent dimensions should be ~4x larger following vicreg
        """
        super().__init__(loss)
        self.model = Projector(dimensions)

    def run_head(
        self,
        backbone_features,
        batch,
        stage,
        save_image=False,
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
