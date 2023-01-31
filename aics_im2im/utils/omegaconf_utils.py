from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from typing import Union


def kv_to_dict(kv: Union[DictConfig, ListConfig]) -> DictConfig:
    if isinstance(kv, DictConfig):
        OmegaConf.resolve(kv)
        # postprocessing
        for k, v in kv.items():
            kv[k] = dict(v)
    elif isinstance(kv, ListConfig):
        # task heads
        ret = {}
        for item in kv:
            assert len(item) == 2, f"Expected ListConfig to have len 2, got {len(item)}"
            ret[item[0]] = OmegaConf.to_container(item[1], resolve=True)
        kv = ret
    else:
        raise TypeError(
            "Config resolved with kv_to_dict must be ListConfig or DictConfig"
        )
    return OmegaConf.create(kv)


def remove_aux_key(cfg):
    with open_dict(cfg):
        for field in cfg.keys():
            if isinstance(cfg[field], DictConfig):
                try:
                    del cfg[field]["_aux"]
                except KeyError:
                    continue
    return cfg
