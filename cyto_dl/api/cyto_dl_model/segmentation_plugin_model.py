from pathlib import Path
from typing import List, Optional, Tuple, Union

from omegaconf import DictConfig, ListConfig

from cyto_dl.api.cyto_dl_model import CytoDLBaseModel
from cyto_dl.api.data import ExperimentType, HardwareType, PatchSize


class SegmentationPluginModel(CytoDLBaseModel):
    """A SegmentationPluginModel handles configuration, training, and prediction using the default
    segmentation_plugin experiment from CytoDL."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._has_split_column = False

    @classmethod
    def _get_experiment_type(cls) -> ExperimentType:
        return ExperimentType.SEGMENTATION_PLUGIN

    def _set_max_epochs(self, max_epochs: int) -> None:
        self._set_cfg("trainer.max_epochs", max_epochs)

    def _set_manifest_path(self, manifest_path: Union[str, Path]) -> None:
        self._set_cfg("data.path", str(manifest_path))

    def _set_output_dir(self, output_dir: Union[str, Path]) -> None:
        self._set_cfg("paths.output_dir", str(output_dir))
        self._set_cfg("paths.work_dir", str(output_dir))

    def set_input_channel(self, input_channel: int) -> None:
        self._set_cfg("input_channel", input_channel)

    def get_input_channel(self) -> int:
        return self._get_cfg("input_channel")

    def set_raw_image_channels(self, channels: int) -> None:
        self._set_cfg("raw_im_channels", channels)

    def get_raw_image_channels(self) -> int:
        return self._get_cfg("raw_im_channels")

    # TODO: is there a better way to deal with column names + split columns?
    def set_manifest_column_names(
        self,
        source: str,
        target1: str,
        target2: str,
        merge_mask: str,
        exclude_mask: str,
        base_image: str,
    ) -> None:
        self._set_cfg("source_col", source)
        self._set_cfg("target_col1", target1)
        self._set_cfg("target_col2", target2)
        self._set_cfg("merge_mask_col", merge_mask)
        self._set_cfg("exclude_mask_col", exclude_mask)
        self._set_cfg("base_image_col", base_image)

    def get_manifest_column_names(self) -> Tuple[str, str, str, str, str, str]:
        return (
            self._get_cfg("source_col"),
            self._get_cfg("target_col1"),
            self._get_cfg("target_col2"),
            self._get_cfg("merge_mask_col"),
            self._get_cfg("exclude_mask_col"),
            self._get_cfg("base_image_col"),
        )

    def set_split_column(self, split_column: str) -> None:
        self._set_cfg("data.split_column", split_column)
        existing_cols: ListConfig = self._get_cfg("data.columns")
        if self._has_split_column:
            existing_cols[-1] = split_column
        else:
            existing_cols.append(split_column)
        self._has_split_column = True

    def get_split_column(self) -> Optional[str]:
        return self._get_cfg("data.split_column")

    def remove_split_column(self) -> None:
        if self._has_split_column:
            self._set_cfg("data.split_column", None)
            existing_cols: ListConfig = self._get_cfg("data.columns")
            del existing_cols[-1]
        self._has_split_column = False

    def set_patch_size(self, patch_size: PatchSize) -> None:
        self._set_cfg("data._aux.patch_shape", patch_size.value)

    def get_patch_size(self) -> Optional[PatchSize]:
        p_shape: ListConfig = self._get_cfg("data._aux.patch_shape")
        return PatchSize(list(p_shape)) if p_shape else None

    def set_hardware_type(self, hardware_type: HardwareType) -> None:
        self._set_cfg("trainer.accelerator", hardware_type.value)

    def get_hardware_type(self) -> HardwareType:
        return HardwareType(self._get_cfg("trainer.accelerator"))
