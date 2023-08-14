import torch
from hydra._internal.utils import _locate
from hydra.utils import instantiate


def load_model_from_checkpoint(ckpt_path, key, model_class, strict=True):
    model_class = _locate(model_class)
    model = model_class.load_from_checkpoint(ckpt_path, strict=strict)
    key = key.split(".")
    for k in key:
        model = model.__getattr__(k)
    return model
