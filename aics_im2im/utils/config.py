from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict, read_write


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


def remove_aux_key(cfg):
    with read_write(cfg):
        for field in cfg.keys():
            if isinstance(cfg[field], DictConfig):
                with read_write(cfg[field]):
                    with open_dict(cfg[field]):
                        try:
                            del cfg[field]["_aux"]
                        except KeyError:
                            continue
    return cfg
