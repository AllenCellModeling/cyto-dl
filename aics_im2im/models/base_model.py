import gc
import inspect
import logging
from typing import Sequence, Union

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.utilities.parsing import get_init_args

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


def _parse_init_args(frame):
    init_args = get_init_args(frame)
    if frame.f_back.f_code.co_name == "__init__":
        # if this was called from a subclass's init
        init_args.update(get_init_args(frame.f_back))

    init_args = {k: _cast_omegaconf(v) for k, v in init_args.items()}
    ignore = [arg for arg, v in init_args.items() if not _is_primitive(v)]
    if "optimizer" in ignore:
        ignore.remove("optimizer")

    if "lr_scheduler" in ignore:
        ignore.remove("lr_scheduler")

    for arg in ignore:
        del init_args[arg]
    return init_args


class BaseModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        frame = inspect.currentframe()
        init_args = _parse_init_args(frame)

        self.save_hyperparameters(init_args, logger=False)
        self.optimizer = init_args.pop("optimizer", torch.optim.Adam)
        self.lr_scheduler = init_args.pop("lr_scheduler", None)
        self.cache_outputs = init_args.get("cache_outputs", ("test",))
        self._cached_outputs = {}

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

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        return self._step("train", batch, batch_idx, logger=True, optimizer_idx=optimizer_idx)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch, batch_idx, logger=True)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch, batch_idx, logger=False)

    def predict_step(self, batch, batch_idx):
        return self._step("predict", batch, batch_idx, logger=False)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "frequency": 1,
                },
            }
        return optimizer
