from omegaconf import OmegaConf, open_dict


def resolve_readonly(cfg):
    with open_dict(cfg):
        cfg._get_child("hydra")._set_flag("readonly", False)
        cfg.hydra._get_child("run")._set_flag("readonly", False)
        cfg.hydra.run._get_child("dir")._set_flag("readonly", False)
        OmegaConf.resolve(cfg)
        cfg._get_child("hydra")._set_flag("readonly", True)
        cfg.hydra._get_child("run")._set_flag("readonly", True)
        cfg.hydra.run._get_child("dir")._set_flag("readonly", True)
