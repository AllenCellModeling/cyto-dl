from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig
from torch.nn.modules.loss import _Loss as Loss

from .base_model import BaseModel

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]


class BasicModel(BaseModel):
    """A minimal Pytorch Lightning wrapper around generic Pytorch models."""

    def __init__(
        self,
        network: Optional[nn.Module] = None,
        loss: Optional[Loss] = None,
        x_label: str = "x",
        y_label: str = "y",
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        squeeze_y: bool = False,
        save_predictions: Optional[Callable] = None,
        fields_to_log: Optional[Sequence] = None,
        pretrained_weights: Optional[str] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        network: Optional[nn.Module] = None
            The network to wrap
        loss: Optional[Loss] = None
            The loss function to optimize for
        x_label: str = "x"
            The key used to retrieve the input from dataloader batches
        y_label: str = "y"
            The key used to retrieve the target from dataloader batches
        optimizer: torch.optim.Optimizer = torch.optim.Adam
            The optimizer class
        save_predictions: Optional[Callable] = None
            A function to save the results of `aics_im2im predict`
        fields_to_log: Optional[Union[Sequence, Dict]] = None
            List of batch fields to store with the outputs. Use a list to log
            the same fields for every training stage (train, val, test, prediction).
            If a list is used, it is assumed to be for test and prediction only
        pretrained_weights: Optional[str] = None
            Path to pretrained weights. If network is not None, this will be
            loaded via `network.load_state_dict`, otherwise it will be
            loaded via `torch.load`.
        """
        super().__init__()

        if network is None and pretrained_weights is None:
            raise ValueError("`network` and `pretrained_weights` can't both be None.")

        if pretrained_weights is not None:
            pretrained_weights = torch.load(pretrained_weights)

        if network is not None:
            self.network = network
            if pretrained_weights is not None:
                self.network.load_state_dict(pretrained_weights)
        else:
            self.network = pretrained_weights

        if loss is not None:
            self.loss = loss
        else:
            self.loss = nn.MSELoss()

        self._squeeze_y = False

        if save_predictions is not None:
            self.save_predictions = save_predictions
        self.fields_to_log = fields_to_log

    def parse_batch(self, batch):
        return (batch[self.hparams.x_label], batch[self.hparams.y_label], {})

    def forward(self, x, **kwargs):
        return self.network(x, **kwargs)

    def _step(self, stage, batch, batch_idx, logger):
        x, y, forward_kwargs = self.parse_batch(batch)

        yhat = self.forward(x, **forward_kwargs)

        if self._squeeze_y:
            loss = self.loss(yhat.squeeze(), y.squeeze())
        else:
            try:
                loss = self.loss(yhat, y)
            except (RuntimeError, ValueError) as err:
                if _check_one_dimensional_shapes(y.shape, yhat.shape):
                    try:
                        loss = self.loss(yhat.squeeze(), y.squeeze())
                        self._squeeze_y = True
                    except Exception as inner_err:
                        raise inner_err from err
                else:
                    raise err

        if stage != "predict":
            self.log(
                f"{stage}_loss",
                loss.detach(),
                logger=logger,
                on_step=True,
                on_epoch=True,
            )

        output = {
            "loss": loss,
            "yhat": yhat.detach().squeeze(),
            "y": y.detach().squeeze(),
        }

        if isinstance(self.fields_to_log, (list, ListConfig)):
            if stage in ("predict", "test"):
                for field in self.fields_to_log:
                    output[field] = batch[field]

        elif isinstance(self.fields_to_log, (dict, DictConfig)):
            if stage in self.fields_to_log:
                for field in self.fields_to_log[stage]:
                    output[field] = batch[field]

        return output


def _check_one_dimensional_shapes(shape1, shape2):
    return (len(shape1) == 1 and len(shape2) == 2 and shape2[-1] == 1) or (
        len(shape2) == 1 and len(shape1) == 2 and shape1[-1] == 1
    )
