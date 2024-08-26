import numpy as np
from omegaconf import OmegaConf


def resolve_readonly(cfg):
    OmegaConf.set_readonly(cfg.hydra, None)
    OmegaConf.resolve(cfg)


def skip_test(test_name):
    """Skip pretraining models for testing."""
    return bool(np.any([x in test_name for x in ("mae", "ijepa", "iwm", "hiera")]))
