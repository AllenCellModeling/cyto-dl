import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from cyto_dl.train import train
from tests.helpers.run_if import RunIf

from .utils import resolve_readonly, skip_test


@pytest.mark.parametrize("spatial_dims", [2, 3])
def test_train_fast_dev_run(cfg_train, spatial_dims, request):
    """Run for 1 train, val and test step."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"

        cfg_train.spatial_dims = spatial_dims
        if spatial_dims == 2:
            cfg_train.data._aux.patch_shape = [16, 16]

        cfg_train.test = not skip_test(request.node.name)
    resolve_readonly(cfg_train)
    train(cfg_train)


@RunIf(min_gpus=1)
@pytest.mark.parametrize("spatial_dims", [2, 3])
def test_train_fast_dev_run_gpu(cfg_train, spatial_dims, request):
    """Run for 1 train, val and test step on GPU."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"

        cfg_train.spatial_dims = spatial_dims
        if spatial_dims == 2:
            cfg_train.data._aux.patch_shape = [16, 16]
        cfg_train.test = not skip_test(request.node.name)

    resolve_readonly(cfg_train)
    train(cfg_train)


@pytest.mark.slow
@pytest.mark.parametrize("spatial_dims", [2, 3])
def test_train_resume(tmp_path, cfg_train, spatial_dims, request):
    """Run 1 epoch, finish, and resume for another epoch."""
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 2
        cfg_train.callbacks.model_checkpoint.monitor = "train/loss"
        cfg_train.callbacks.model_checkpoint.save_top_k = 2

        cfg_train.spatial_dims = spatial_dims
        if spatial_dims == 2:
            cfg_train.data._aux.patch_shape = [16, 16]
        cfg_train.test = not skip_test(request.node.name)

    HydraConfig().set_config(cfg_train)

    resolve_readonly(cfg_train)
    metric_dict_1, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "last.ckpt" in files
    assert "epoch_000.ckpt" in files

    with open_dict(cfg_train):
        cfg_train.checkpoint.ckpt_path = str(tmp_path / "checkpoints" / "epoch_000.ckpt")
    metric_dict_2, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "epoch_001.ckpt" in files
    assert "epoch_002.ckpt" not in files
