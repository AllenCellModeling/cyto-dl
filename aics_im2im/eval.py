import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` above is optional line to make environment more convenient
# should be placed at the top of each entry file
#
# main advantages:
# - allows you to keep all entry files in "aics_im2im/" without installing project as a package
# - launching python file works no matter where is your current work dir
# - automatically loads environment variables from ".env" if exists
#
# how it works:
# - `setup_root()` above recursively searches for either ".git" or "pyproject.toml" in present
#   and parent dirs, to determine the project root dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can be run from
#   any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
#   to make all paths always relative to project root
# - loads environment variables from ".env" in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project root dir
# 2. remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from typing import List, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from aics_im2im import utils

log = utils.get_pylogger(__name__)
from omegaconf import DictConfig, ListConfig, OmegaConf


def kv_to_dict(kv: ListConfig) -> DictConfig:
    """
    Parameters
    ----------
    kv: ListConfig
        ListConfig where every item is a nested list of the form [interpolated key, value]
    Returns
    -------
    DictConfig of input
    """
    if isinstance(kv, ListConfig):
        ret = {}
        for item in kv:
            assert (
                len(item) == 2
            ), f"Expected ListConfig item to have len 2, got {len(item)}"
            ret[item[0]] = OmegaConf.to_container(item[1], resolve=True)
    else:
        raise TypeError("Config resolved with kv_to_dict must be ListConfig")
    return OmegaConf.create(ret)

@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    if cfg.get("mode", "predict") == "test":
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    else:
        # for predictions use trainer.predict(...)
        trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    OmegaConf.register_new_resolver("kv_to_dict", kv_to_dict)
    OmegaConf.register_new_resolver("eval", eval)
    evaluate(cfg)


if __name__ == "__main__":
    main()
