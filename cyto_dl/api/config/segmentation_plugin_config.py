from typing import Tuple, List, Union
from pathlib import Path
from cyto_dl.api.config import CytoDLConfig
from cyto_dl.api.data import *

class SegmentationPluginConfig(CytoDLConfig):
    
    def _get_experiment_type(self) -> ExperimentType:
        return ExperimentType.SEGMENTATION_PLUGIN
    
    # a lot of these keys are duplicated across the im2im experiment types (but not present in top-level
    # train.yaml or eval.yaml) - should we move these into the top-level configs and move these setters and
    # getters accordingly? 
    def set_spatial_dims(self, spatial_dims: int) -> None:
        self._set_cfg("spatial_dims", spatial_dims)
    
    def get_spatial_dims(self) -> int:
        return self._get_cfg("spatial_dims")
    
    def set_input_channel(self, input_channel: int) -> None:
        self._set_cfg("input_channel", input_channel)
    
    def get_input_channel(self) -> int:
        return self._get_cfg("input_channel")
    
    def set_raw_image_channels(self, channels: int) -> None:
        self._set_cfg("raw_im_channels", channels)
    
    def get_raw_image_channels(self) -> int:
        return self._get_cfg("raw_im_channels")
    
    def set_data_path(self, data_path: Union[str, Path]) -> None:
        self._set_cfg("data.path", str(data_path))
    
    def get_data_path(self) -> str:
        return self._get_cfg("data.path")
    
    # TODO: is there a better way to deal with column names + split columns?
    def set_manifest_column_names(self, source: str, target1: str, target2: str, merge_mask: str, exclude_mask: str, base_image: str) -> None:
        self._set_cfg("source_col", source)
        self._set_cfg("target_col1", target1)
        self._set_cfg("target_col2", target2)
        self._set_cfg("merge_mask_col", merge_mask)
        self._set_cfg("exclude_mask_col", exclude_mask)
        self._set_cfg("base_image_col", base_image)
    
    def get_manifest_column_names(self) -> Tuple[str, str, str, str, str, str]:
        return self._get_cfg("source_col"), self._get_cfg("target_col1"), self._get_cfg("target_col2"), self._get_cfg("merge_mask_col"), self._get_cfg("exclude_mask_col"), self._get_cfg("base_image_col")
    
    def set_split_column(self, split_column: str) -> None:
        self._set_cfg("data.split_column", split_column)
        # need to also add it to the list of columns
        existing_cols: List[str] = self._get_cfg("data.columns")
        if len(existing_cols) > len(self.get_manifest_column_names()):
            existing_cols[-1] = split_column
        else:
            existing_cols.append(split_column)
    
    def get_split_column(self) -> str:
        self._get_cfg("data.split_column")
    
    def set_hardware_type(self, hardware_type: HardwareType) -> None:
        self._set_cfg("trainer.accelerator", hardware_type.value)
    
    def get_hardware_type(self) -> HardwareType:
        return HardwareType(self._get_cfg("trainer.accelerator"))

    def set_max_epochs(self, max_epochs: int) -> None:
        self._set_cfg("trainer.max_epochs", max_epochs)
    
    def get_max_epochs(self) -> int:
        return self._get_cfg("trainer.max_epochs")
    
    def set_output_dir(self, output_dir: Union[str, Path]) -> None:
        self._set_cfg("paths.output_dir", str(output_dir))
    
    def get_output_dir(self) -> str:
        return self._get_cfg("paths.output_dir")
    
    # I can't find where this is actually used in cyto_dl, do we need to support this?
    def set_work_dir(self, work_dir: Union[str, Path]) -> None:
        self._set_cfg("paths.work_dir", str(work_dir))
    
    def get_work_dir(self) -> str:
        return self._get_cfg("paths.work_dir")
