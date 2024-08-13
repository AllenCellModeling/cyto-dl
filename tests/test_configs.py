import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from cyto_dl.utils.config import remove_aux_key

from .utils import resolve_readonly


@pytest.mark.parametrize("spatial_dims", [2, 3])
def test_train_config(cfg_train: DictConfig, spatial_dims: int):
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    with open_dict(cfg_train):
        cfg_train.spatial_dims = spatial_dims
        if spatial_dims == 2:
            cfg_train.data._aux.patch_shape = [16, 16]

    HydraConfig().set_config(cfg_train)
    resolve_readonly(cfg_train)
    remove_aux_key(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


@pytest.mark.parametrize("spatial_dims", [2, 3])
def test_eval_config(cfg_eval: DictConfig, spatial_dims: int):
    assert cfg_eval
    assert cfg_eval.data
    assert cfg_eval.model
    assert cfg_eval.trainer

    with open_dict(cfg_eval):
        cfg_eval.spatial_dims = spatial_dims
        if spatial_dims == 2:
            cfg_eval.data._aux.patch_shape = [16, 16]

    HydraConfig().set_config(cfg_eval)
    resolve_readonly(cfg_eval)
    remove_aux_key(cfg_eval)

    hydra.utils.instantiate(cfg_eval.data)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)
