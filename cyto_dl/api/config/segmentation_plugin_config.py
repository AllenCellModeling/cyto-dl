from typing import Optional
from pathlib import Path
from cyto_dl.api.config import CytoDLConfig
from cyto_dl.api.data import *

class SegmentationPluginConfig(CytoDLConfig):
    def __init__(self, config_filepath: Optional[Path]=None):
        super().__init__(config_filepath)
    
    def get_experiment_type(self) -> ExperimentType:
        return ExperimentType.SEGMENTATION_PLUGIN