from omegaconf import OmegaConf, open_dict


def resolve_readonly(cfg):
    OmegaConf.set_readonly(cfg.hydra, None)
    OmegaConf.resolve(cfg)
