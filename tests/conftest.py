from typing import Generator

import pyrootutils
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict

from aics_im2im.utils.config import kv_to_dict

OmegaConf.register_new_resolver("kv_to_dict", kv_to_dict)
OmegaConf.register_new_resolver("eval", eval)

# Experiment configs to test
experiments = [
    {
        "exp_path": "im2im/segmentation.yaml",
        "data_path": "test/segmentation.yaml",
    },  # Segmentation
    {"exp_path": "im2im/omnipose.yaml", "data_path": "test/omnipose.yaml"},  # Omnipose
]


@pytest.fixture(scope="package", params=experiments)
def cfg_train_global(request) -> DictConfig:
    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                f"experiment={request.param['exp_path']}",
                f"data={request.param['data_path']}",
                "trainer=cpu.yaml",
            ],
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root())
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None
    return cfg


@pytest.fixture(scope="package", params=experiments)
def cfg_eval_global(request) -> DictConfig:
    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=[
                "ckpt_path=.",
                f"experiment={request.param['exp_path']}",
                f"data={request.param['data_path']}",
                "trainer=cpu.yaml",
            ],
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root())
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None
    return cfg


# this is called by each test which uses `cfg_train` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_train(cfg_train_global, tmp_path) -> Generator[DictConfig, None, None]:
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


# this is called by each test which uses `cfg_eval` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global, tmp_path) -> Generator[DictConfig, None, None]:
    cfg = cfg_eval_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()
