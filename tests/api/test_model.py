from unittest.mock import patch

import pytest

import cyto_dl.api.model
from cyto_dl.api.data import ExperimentType
from cyto_dl.api.model import CytoDLModel


# mock these functions to avoid attempts to write to file system
@patch("cyto_dl.api.model.OmegaConf.save")
@patch("cyto_dl.api.model.Path.mkdir")
def test_load_default_experiment_valid_exp_type(MockMkdir, MockSave):
    model: CytoDLModel = CytoDLModel()
    model.load_default_experiment(ExperimentType.SEGMENTATION.value, "fake_dir")
    MockMkdir.assert_called()
    MockSave.assert_not_called()


@patch("cyto_dl.api.model.OmegaConf.save")
@patch("cyto_dl.api.model.Path.mkdir")
def test_load_default_experiment_invalid_exp_type(MockMkdir, MockSave):
    model: CytoDLModel = CytoDLModel()
    with pytest.raises(AssertionError):
        model.load_default_experiment("invalid_exp_type", "fake_dir")
    MockMkdir.assert_not_called()
    MockSave.assert_not_called()
