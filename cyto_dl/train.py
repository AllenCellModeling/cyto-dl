import os
from collections.abc import MutableMapping
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple

import hydra
import lightning
import pyrootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.tuner import Tuner
from omegaconf import DictConfig, OmegaConf

from cyto_dl import utils

log = utils.get_pylogger(__name__)

with suppress(ValueError):
    OmegaConf.register_new_resolver("kv_to_dict", utils.kv_to_dict)
    OmegaConf.register_new_resolver("eval", eval)


@utils.task_wrapper
def train(cfg: DictConfig, data=None) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    # set root if not already setup
    if os.environ.get("PROJECT_ROOT") is None:
        root = pyrootutils.setup_root(
            search_from=__file__,
            project_root_env_var=True,
            dotenv=True,
            pythonpath=True,
            cwd=False,  # do NOT change working directory to root (would cause problems in DDP mode)
        )
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        lightning.seed_everything(cfg.seed, workers=True)

    # resolve config to avoid unresolvable interpolations in the stored config

    OmegaConf.resolve(cfg)

    # remove aux section after resolving and before instantiating
    utils.remove_aux_key(cfg)

    log.info(f"Instantiating data <{cfg.data.get('_target_', cfg.data)}>")

    use_batch_tuner = False
    if cfg.data.get("batch_size") == "AUTO":
        use_batch_tuner = True
        cfg.data.batch_size = 1

    data = utils.create_dataloader(cfg.data, data)
    if not isinstance(data, LightningDataModule):
        if not isinstance(data, MutableMapping) or "train_dataloaders" not in data:
            raise ValueError(
                "`data` config for training must be either a LightningDataModule "
                "or a dict with keys `train_dataloaders` and (optionally) "
                "`val_dataloaders`, specifying train and validation DataLoaders "
                "(or lists thereof)."
            )

    log.info(f"Instantiating model <{cfg.model._target_}>")

    model: LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "data": data,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if use_batch_tuner:
        tuner = Tuner(trainer=trainer)
        tuner.scale_batch_size(model, datamodule=data, mode="power")

    if cfg.get("train"):
        log.info("Starting training!")
        model, load_params = utils.load_checkpoint(model, cfg.get("checkpoint"))
        if isinstance(data, LightningDataModule):
            trainer.fit(model=model, datamodule=data, ckpt_path=load_params.get("ckpt_path"))
        else:
            trainer.fit(
                model=model,
                train_dataloaders=data.train_dataloaders,
                val_dataloaders=data.val_dataloaders,
                ckpt_path=load_params.get("ckpt_path"),
            )

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        if not isinstance(data, LightningDataModule):
            log.warning(
                "To test after training, `data` must be a LightningDataModule. Skipping testing."
            )
        else:
            log.info("Starting testing!")
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if ckpt_path == "":
                log.warning("Best ckpt not found! Using current weights for testing...")
                ckpt_path = None
            trainer.test(model=model, datamodule=data, ckpt_path=ckpt_path)
            log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(
    version_base="1.3",
    config_path=os.environ.get("CYTODL_CONFIG_PATH", "../configs"),
    config_name="train.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    if cfg.get("persist_cache", False) or cfg.data.get("cache_dir") is None:
        metric_dict, _ = train(cfg)
    else:
        Path(cfg.data.cache_dir).mkdir(exist_ok=True, parents=True)
        with TemporaryDirectory(dir=cfg.data.cache_dir) as temp_dir:
            cfg.data.cache_dir = temp_dir
            metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
