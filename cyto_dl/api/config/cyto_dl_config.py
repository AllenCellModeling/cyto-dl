from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
from cyto_dl.api.data import *


class CytoDLConfig(ABC):
    """
    A CytoDLConfig represents the configuration for a CytoDLModel. Can be
    passed to CytoDLModel on initialization to create a fully-functional model.
    """
    # Note: internally, Config classes will just be used to store overrides and
    # possibly a config filepath, while CytoDLModel will handle actually generating
    # the full config
    @abstractmethod
    def __init__(self, config_filepath: Optional[Path]=None, train: bool=True):
        """
        :param config_filepath: path to a .yaml config file that will be used as the basis
        for this CytoDLConfig. If None, a default config will be used instead.
        :param train: indicates whether this config will be used for training or prediction
        """
        self._config_filepath: str = config_filepath
        self._overrides: Dict[str, Any] = {}
        self._overrides["train"] = train
        self._overrides["test"] = train
    
    @abstractmethod
    def get_experiment_type(self) -> ExperimentType:
        """
        Return experiment type for this config (e.g. segmentation_plugin, gan, etc)
        """
        pass

    def get_config_filepath(self) -> Optional[Path]:
        return self._config_filepath
    
    def set_hardware_type(self, hardware_type: HardwareType) -> None:
        self._overrides["trainer.accelerator"] = hardware_type.value
    
    def set_experiment_name(self, name: str) -> None:
        self._overrides["experiment_name"] = name
    
    def set_ckpt_path(self, ckpt_path: Path) -> None:
        self._overrides["ckpt_path"] = str(ckpt_path.resolve())

