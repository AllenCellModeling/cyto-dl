from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Any
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict, DictConfig
import pyrootutils
from copy import deepcopy
from cyto_dl.api.data import *


class CytoDLConfig(ABC):
    """
    A CytoDLConfig represents the configuration for a CytoDLModel. Can be
    passed to CytoDLModel on initialization to create a fully-functional model.
    """
    def __init__(self, config_filepath: Optional[Path]=None, train: bool=True):
        """
        :param config_filepath: path to a .yaml config file that will be used as the basis
        for this CytoDLConfig. If None, a default config will be used instead.
        :param train: indicates whether this config will be used for training or prediction
        """
        self._config_filepath: str = config_filepath
        self._train: bool = train
        self._cfg: DictConfig = OmegaConf.load(config_filepath) if config_filepath else self._generate_default_config()
        self._set_cfg("train", train)
        self._set_cfg("test", train)
        # afaik, task_name isn't used outside of template_utils.py - do we need to support this?
        self._set_cfg("task_name", "train" if train else "predict")

        # we currently have an override for ['mode'] in ml-seg, but I can't find top-level 'mode' in the configs, 
        # do we need to support this?
    
    def _generate_default_config(self) -> DictConfig:
        cfg_dir: Path = pyrootutils.find_root(search_from=__file__, indicator=("pyproject.toml", "README.md")) / "configs"
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.2", config_dir=str(cfg_dir)):
            cfg: DictConfig = compose(
                config_name="train.yaml", # only using train.yaml after conversation w/ Benji
                return_hydra_config=True,
                overrides=[f"experiment=im2im/{self._get_experiment_type().name}"],
            )
        with open_dict(cfg):
            del cfg["hydra"]
            cfg.extras.enforce_tags = False
            cfg.extras.print_config = False
        return cfg

    @abstractmethod
    def _get_experiment_type(self) -> ExperimentType:
        """
        Return experiment type for this config (e.g. segmentation_plugin, gan, etc)
        """
        pass
    
    def _key_exists(self, k: str) -> bool:
        keys: List[str] = k.split(".")
        curr_dict: DictConfig = self._cfg
        while keys:
            key: str = keys.pop(0)
            if not key in curr_dict:
                return False
            curr_dict = curr_dict[key]
        return True
    
    def _set_cfg(self, k: str, v: Any) -> None:
        if not self._key_exists(k):
            raise KeyError(f"{k} not found in config dict")
        OmegaConf.update(self._cfg, k, v)

    def _get_cfg(self, k: str) -> Any:
        if not self._key_exists(k):
            raise KeyError(f"{k} not found in config dict")
        return OmegaConf.select(self._cfg, k)
    
    def set_experiment_name(self, name: str) -> None:
        self._set_cfg("experiment_name", name)
    
    def get_experiment_name(self) -> str:
        return self._get_cfg("experiment_name")

    def set_ckpt_path(self, ckpt_path: Path) -> None:
        self._set_cfg("ckpt_path", str(ckpt_path.resolve()))
    
    def get_ckpt_path(self) -> Path:
        return Path(self._get_cfg("ckpt_path"))
    
    def get_config(self) -> DictConfig:
        return deepcopy(self._cfg)


