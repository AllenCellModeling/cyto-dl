from pathlib import Path

import numpy as np
import pytest

from cyto_dl.api import CytoDLModel
from cyto_dl.utils import extract_array_predictions


@pytest.mark.skip
def test_array_train(tmp_path):
    model = CytoDLModel()

    overrides = {
        "trainer.max_epochs": 1,
        "logger": None,
        "trainer.accelerator": "cpu",
        "trainer.devices": 1,
    }

    model.load_default_experiment(experiment_type="segmentation_array", output_dir=tmp_path)
    model.override_config(overrides)

    data = {
        "train": [{"raw": np.random.randn(1, 40, 256, 256), "seg": np.ones((1, 40, 256, 256))}],
        "val": [{"raw": np.random.randn(1, 40, 256, 256), "seg": np.ones((1, 40, 256, 256))}],
    }
    model.train(data=data)

    ckpt_dir = Path(model.cfg.callbacks.model_checkpoint.dirpath)
    assert "last.ckpt" in [fn.name for fn in ckpt_dir.iterdir()]
    return ckpt_dir / "last.ckpt"


@pytest.mark.slow
def test_array_train_predict(tmp_path):
    ckpt_path = test_array_train(tmp_path)

    model = CytoDLModel()

    overrides = {
        "logger": None,
        "trainer.accelerator": "cpu",
        "trainer.devices": 1,
        "checkpoint.ckpt_path": ckpt_path,
    }

    model.load_default_experiment(
        experiment_type="segmentation_array",
        output_dir=tmp_path,
        train=False,
        overrides=["data=im2im/numpy_dataloader_predict"],
    )
    model.override_config(overrides)
    model.print_config()

    data = [np.random.rand(1, 32, 64, 64), np.random.rand(1, 32, 64, 64)]
    _, _, output = model.predict(data=data)
    preds = extract_array_predictions(output)

    for head in model.cfg.model.task_heads.keys():
        assert preds[head].shape[0] == len(data)
        assert preds[head].shape[1:] == data[0].shape
