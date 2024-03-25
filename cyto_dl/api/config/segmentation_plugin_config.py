from typing import Optional, Tuple, List
from pathlib import Path
from cyto_dl.api.config import CytoDLConfig
from cyto_dl.api.data import *
# TODO: get rid of using Path or any non-primitive data type since there are format strings in config sometimes...
class SegmentationPluginConfig(CytoDLConfig):
    def __init__(self, config_filepath: Optional[Path] = None, train: bool = True):
        super().__init__(config_filepath, train)
    
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
    
    def set_data_path(self, data_path: Path) -> None:
        self._set_cfg("data.path", str(data_path))
    
    def get_data_path(self) -> Path:
        return Path(self._get_cfg("data.path"))
    
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
