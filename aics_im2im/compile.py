import datetime
import os
import tempfile
from collections.abc import MutableMapping
from contextlib import suppress
from pathlib import Path
from typing import Tuple

import hydra
from lightning import LightningDataModule
from model_archiver.model_archiver import package_model
from model_archiver.model_packaging_utils import ModelExportUtils
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader

from aics_im2im import utils


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")


log = utils.get_pylogger(__name__)


@utils.task_wrapper
def compile(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    if not cfg.ckpt_path:
        raise ValueError("Checkpoint path must be included for compiling")

    # resolve config to avoid unresolvable interpolations in the stored config
    OmegaConf.resolve(cfg)

    # remove aux section after resolving and before instantiating
    utils.remove_aux_key(cfg)

    data = hydra.utils.instantiate(cfg.data)
    if not isinstance(data, (LightningDataModule, DataLoader)):
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

    pkg_root = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = Path(tmp_dir) / "config.yaml"
        OmegaConf.save(config=cfg, f=cfg_path)

        name = cfg.model._target_ + "_" + cfg.get("run_name", timestamp())

        args = OmegaConf.create(
            {
                "model_file": str(Path(pkg_root) / cfg.model_file),
                "handler": str(Path(pkg_root) / cfg.handler_file),
                "serialized_file": cfg.ckpt_path,
                "model_name": name,
                "extra_files": [cfg_path],
                "export_file_path": cfg.get("export_file_path", Path.cwd() / f"{name}.mar"),
            }
        )
        manifest = ModelExportUtils.generate_manifest_json(args)
        package_model(args, manifest=manifest)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    with suppress(ValueError):
        OmegaConf.register_new_resolver("kv_to_dict", utils.kv_to_dict)
        OmegaConf.register_new_resolver("eval", eval)
    compile(cfg)


if __name__ == "__main__":
    main()
