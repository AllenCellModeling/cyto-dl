import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from aics_im2im.eval import evaluate
from aics_im2im.train import train

from .conftest import experiment_types
from .utils import resolve_readonly


@pytest.mark.slow
@pytest.mark.parametrize(
    "cfg_train_global, cfg_eval_global",
    zip(experiment_types, experiment_types),
    indirect=True,
)
def test_train_eval(tmp_path, cfg_train, cfg_eval):
    """Train for 1 epoch with `train.py` and evaluate with `eval.py`"""
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1

    HydraConfig().set_config(cfg_train)
    resolve_readonly(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_eval.test = True

    HydraConfig().set_config(cfg_eval)
    resolve_readonly(cfg_eval)
    test_metric_dict, _ = evaluate(cfg_eval)

    assert test_metric_dict["test_loss"] > 0.0
    assert (
        abs(train_metric_dict["test_loss"].item() - test_metric_dict["test_loss"].item()) < 0.001
    )
