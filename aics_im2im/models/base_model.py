import inspect
import logging
from collections.abc import MutableMapping
from typing import Optional, Sequence, Union

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

Array = Union[torch.Tensor, np.ndarray, Sequence[float]]
logger = logging.getLogger("lightning")
logger.propagate = False


def _is_primitive(value):
    if isinstance(value, (type(None), bool, str, int, float)):
        return True

    if isinstance(value, (tuple, list)):
        return all(_is_primitive(el) for el in value)

    if isinstance(value, dict):
        return all(_is_primitive(el) for el in value.values())

    return False


def _cast_omegaconf(value):
    if isinstance(value, (ListConfig, DictConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


class BaseModelMeta(type):
    def __call__(cls, *args, **kwargs):
        """Meta class to handle instantiating configs."""

        init_args = inspect.signature(cls.__init__).parameters.copy()
        init_args.pop("self")
        keys = tuple(init_args.keys())

        init_args = {keys[ix]: arg for ix, arg in enumerate(args)}
        init_args.update(kwargs)

        # instantiate class with instantiated `init_args`
        # hydra doesn't change the original dict, so we can use it after this
        # with `save_hyperparameters`
        obj = type.__call__(cls, **instantiate(init_args, _recursive_=True, _convert_=True))

        # make sure only primitives get stored in the ckpt
        init_args = {k: _cast_omegaconf(v) for k, v in init_args.items()}
        ignore = [arg for arg, v in init_args.items() if not _is_primitive(v)]

        for arg in ignore:
            init_args.pop(arg)

        obj.save_hyperparameters(init_args, logger=False)

        return obj


class BaseModel(pl.LightningModule, metaclass=BaseModelMeta):
    def __init__(
        self,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        super().__init__()

        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam
        self.lr_scheduler = lr_scheduler

    def parse_batch(self, batch):
        raise NotImplementedError

    def forward(self, x, **kwargs):
        raise NotImplementedError

    def _step(self, stage, batch, batch_idx, logger):
        """Here you should implement the logic for a step in the training/validation/test process.
        The stage (training/validation/test) is given by the variable `stage`.

        Example:

        x = self.parse_batch(batch)

        if self.hparams.id_label is not None:
           if self.hparams.id_label in batch:
               ids = batch[self.hparams.id_label].detach().cpu()
               results.update({
                   self.hparams.id_label: ids,
                   "id": ids
               })

        return results
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx, logger=True)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch, batch_idx, logger=True)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch, batch_idx, logger=False)

    def predict_step(self, batch, batch_idx):
        return self._step("predict", batch, batch_idx, logger=False)

    def configure_optimizers(self):
        optimizer_cls = self.hparams.get("optimizer", torch.optim.Adam)
        optimizer = optimizer_cls(self.parameters())

        scheduler_cls = self.hparams.get("lr_scheduler")
        if scheduler_cls is not None:
            scheduler = scheduler_cls(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "frequency": 1,
                },
            }
        return optimizer
