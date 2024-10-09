from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig
from torch.nn.modules.loss import _Loss as Loss
from torchmetrics import MeanMetric

from .base_model import BaseModel

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]


class BasicModel(BaseModel):
    """A minimal Pytorch Lightning wrapper around generic Pytorch models."""

    def __init__(
        self,
        network: Optional[nn.Module] = None,
        loss: Optional[Loss] = None,
        x_label: str = "x",
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
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
            Assumes that the network outputs gt and predictions
        loss: Optional[Loss] = None
            The loss function to optimize for
        x_label: str = "x"
            The key used to retrieve the input from dataloader batches
        optimizer: torch.optim.Optimizer = torch.optim.Adam
            The optimizer class
        save_predictions: Optional[Callable] = None
            A function to save the results of `serotiny predict`
        fields_to_log: Optional[Union[Sequence, Dict]] = None
            List of batch fields to store with the outputs. Use a list to log
            the same fields for every training stage (train, val, test, prediction).
            If a list is used, it is assumed to be for test and prediction only
        pretrained_weights: Optional[str] = None
            Path to pretrained weights. If network is not None, this will be
            loaded via `network.load_state_dict`, otherwise it will be
            loaded via `torch.load`.
        """

        _DEFAULT_METRICS = {
            "train/loss": MeanMetric(),
            "val/loss": MeanMetric(),
            "test/loss": MeanMetric(),
        }
        metrics = kwargs.pop("metrics", _DEFAULT_METRICS)

        super().__init__(metrics=metrics)

        if network is pretrained_weights is None:
            raise ValueError("`network` and `pretrained_weights` can't both be None.")

        if pretrained_weights is not None:
            pretrained_weights = torch.load(pretrained_weights)  # nosec B614

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

        if save_predictions is not None:
            self.save_predictions = save_predictions
        self.fields_to_log = fields_to_log
        self.x_label = x_label

    def forward(self, x, **kwargs):
        return self.network(x, **kwargs)

    def model_step(self, stage, batch, batch_idx):
        rec, gt = self.forward(batch[self.x_label])
        loss = self.loss(rec, gt).mean()

        output = {
            "loss": loss,
        }

        return output, None, None
