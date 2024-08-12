import os
from collections.abc import MutableMapping
from contextlib import suppress
from typing import List, Tuple

import hydra
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from monai.data import DataLoader as MonaiDataLoader
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader as TorchDataLoader

from cyto_dl import utils

log = utils.get_pylogger(__name__)

with suppress(ValueError):
    OmegaConf.register_new_resolver("kv_to_dict", utils.kv_to_dict)
    OmegaConf.register_new_resolver("eval", eval)


@utils.task_wrapper
def evaluate(cfg: DictConfig, data=None) -> Tuple[dict, dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    if not cfg.checkpoint.ckpt_path:
        raise ValueError("Checkpoint path must be included for testing")

    # resolve config to avoid unresolvable interpolations in the stored config
    OmegaConf.resolve(cfg)

    # remove aux section after resolving and before instantiating
    utils.remove_aux_key(cfg)
    data = utils.create_dataloader(cfg.data, data)
    if not isinstance(data, (LightningDataModule, TorchDataLoader, MonaiDataLoader)):
        if isinstance(data, MutableMapping) and not data.dataloaders:
            raise ValueError(
                "If the `data` config for eval/prediction is a dict it must have a "
                "key `dataloaders` with the corresponding value being a DataLoader "
                "(or list thereof)."
            )
        elif not isinstance(data, (list, ListConfig)):
            raise ValueError(
                "`data` config for eval/prediction must be either:\n"
                " - a LightningDataModule\n"
                " - a DataLoader (or list thereof)\n"
                " - a dict with key `dataloaders`, with the corresponding value "
                "being a DataLoader (or list thereof)"
            )

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    object_dict = {
        "cfg": cfg,
        "data": data,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    model, load_params = utils.load_checkpoint(model, cfg.get("checkpoint"))

    log.info("Starting testing!")
    method = trainer.test if cfg.get("test", False) else trainer.predict
    output = method(model=model, dataloaders=data, ckpt_path=load_params.ckpt_path)
    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict, output


@hydra.main(
    version_base="1.3",
    config_path=os.environ.get("CYTODL_CONFIG_PATH", "../configs"),
    config_name="eval.yaml",
)
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
