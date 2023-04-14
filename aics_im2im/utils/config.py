from contextlib import suppress

from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict, read_write
from omegaconf.errors import InterpolationToMissingValueError, MissingMandatoryValue


def kv_to_dict(kv: ListConfig) -> DictConfig:
    """
    Parameters
    ----------
    kv: ListConfig
        ListConfig where every item is a nested list of the
        form [interpolated key, value]

    Returns
    -------
    DictConfig of input
    """
    if isinstance(kv, ListConfig):
        ret = {}
        for item in kv:
            assert len(item) == 2, f"Expected ListConfig item to have len 2, got {len(item)}"
            ret[item[0]] = OmegaConf.to_container(item[1], resolve=True)
    else:
        raise TypeError("Config resolved with kv_to_dict must be ListConfig")
    return OmegaConf.create(ret)


def is_config(cfg):
    return isinstance(cfg, (ListConfig, DictConfig))


def remove_aux_key(cfg):
    if not is_config(cfg):
        return

    with read_write(cfg):
        with open_dict(cfg):
            if isinstance(cfg, DictConfig):
                for k, v in cfg.items():
                    with suppress(MissingMandatoryValue, InterpolationToMissingValueError):
                        if isinstance(v, DictConfig):
                            remove_aux_key(v)
                            if k == "_aux":
                                del cfg[k]

                        if isinstance(v, ListConfig):
                            for item in v:
                                remove_aux_key(item)
            else:
                for v in cfg:
                    with suppress(MissingMandatoryValue, InterpolationToMissingValueError):
                        if isinstance(v, (DictConfig, ListConfig)):
                            remove_aux_key(v)
