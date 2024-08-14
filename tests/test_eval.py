import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from cyto_dl.eval import evaluate
from cyto_dl.train import train

from .conftest import experiment_types
from .utils import resolve_readonly, skip_test


@pytest.mark.slow
@pytest.mark.parametrize("spatial_dims", [2, 3])
@pytest.mark.parametrize(
    "cfg_train_global, cfg_eval_global",
    zip(experiment_types, experiment_types),
    indirect=True,
)
def test_train_eval(tmp_path, cfg_train, cfg_eval, spatial_dims, request):
    """Train for 1 epoch with `train.py` and evaluate with `eval.py`"""
    if skip_test(request.node.name):
        pytest.skip(f"Skipping {request.node.name}")
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1

        cfg_train.spatial_dims = spatial_dims
        if spatial_dims == 2:
            cfg_train.data._aux.patch_shape = [16, 16]

    HydraConfig().set_config(cfg_train)
    resolve_readonly(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with open_dict(cfg_eval):
        cfg_eval.checkpoint.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_eval.test = True

        cfg_eval.spatial_dims = spatial_dims
        if spatial_dims == 2:
            cfg_eval.data._aux.patch_shape = [16, 16]

    HydraConfig().set_config(cfg_eval)
    resolve_readonly(cfg_eval)
    test_metric_dict, _, _ = evaluate(cfg_eval)

    assert test_metric_dict["test/loss"] > 0.0
    assert (
        abs(train_metric_dict["test/loss"].item() - test_metric_dict["test/loss"].item()) < 0.001
    )
