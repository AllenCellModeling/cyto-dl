from omegaconf import DictConfig, open_dict


def remove_aux_key(cfg):
    with open_dict(cfg):
        for field in cfg.keys():
            if isinstance(cfg[field], DictConfig):
                try:
                    del cfg[field]["_aux"]
                except KeyError:
                    continue
    return cfg
