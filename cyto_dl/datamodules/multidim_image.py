from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union

import pandas as pd
import tqdm
from bioio import BioImage
from monai.data import CacheDataset
from omegaconf import OmegaConf


class MultiDimImageDataset(CacheDataset):
    """Dataset converting a `.csv` file or dictionary listing multi dimensional (timelapse or
    multi-scene) files and some metadata into batches of metadata intended for the
    BioIOImageLoaderd class."""

    def __init__(
        self,
        csv_path: Optional[Union[Path, str]] = None,
        img_path_column: str = "path",
        channel_column: str = "channel",
        spatial_dims: int = 3,
        scene_column: str = "scene",
        resolution_column: str = "resolution",
        time_start_column: str = "start",
        time_stop_column: str = "stop",
        time_step_column: str = "step",
        extra_columns: Sequence[str] = [],
        dict_meta: Optional[Dict] = None,
        transform: Optional[Union[Callable, Sequence[Callable]]] = [],
        **cache_kwargs,
    ):
        """
        Parameterss
        ----------
        csv_path: Union[Path, str]
            path to csv
        img_path_column: str
            column in `csv_path` that contains path to multi dimensional (timelapse or multi-scene) file
        channel_column:str
            Column in `csv_path` that contains which channel to extract from multi dimensional (timelapse or multi-scene) file. Should be an integer.
        spatial_dims:int=3
            Spatial dimension of output image. Must be 2 for YX or 3 for ZYX. Spatial dimensions are used to specify the dimension order of the output image, which will be in the format `CZYX` or `CYX` to ensure compatibility with dictionary-based MONAI-style transforms.
        scene_column:str="scene",
            Column in `csv_path` that contains scenes to extract from multi-scene file. If not specified, all scenes will
            be extracted. If multiple scenes are specified, they should be separated by a comma (e.g. `scene1,scene2`)
        resolution_column:str="resolution"
            Column in `csv_path` that contains resolution to extract from multi-resolution file. If not specified, resolution is assumed to be 0.
        time_start_column:str="start"
            Column in `csv_path` specifying which timepoint in timelapse image to start extracting. If any of `start_column`, `stop_column`, or `step_column`
            are not specified, all timepoints are extracted.
        time_stop_column:str="stop"
            Column in `csv_path` specifying which timepoint in timelapse image to stop extracting. If any of `start_column`, `stop_column`, or `step_column`
            are not specified, all timepoints are extracted.
        time_step_column:str="step"
            Column in `csv_path` specifying step between timepoints. For example, values in this column should be `2` if every other timepoint should be run.
            If any of `start_column`, `stop_column`, or `step_column` are not specified, all timepoints are extracted.
        extra_columns: Sequence[str] = []
            List of extra columns to include in the output dictionary. These columns will be added to the output dictionary as-is if found, otherwise their value
            will be set to None.
        dict_meta: Optional[Dict]
            Dictionary version of CSV file. If not provided, CSV file is read from `csv_path`.
        transform: Optional[Callable] = []
            List (or Compose Object) or Monai dictionary-style transforms to apply to the image metadata. Typically, the first transform should be BioIOImageLoaderd.
        cache_kwargs:
            Additional keyword arguments to pass to `CacheDataset`. To skip the caching mechanism, set `cache_num` to 0.
        """
        df = (
            pd.read_csv(csv_path)
            if csv_path is not None
            else pd.DataFrame(OmegaConf.to_container(dict_meta))
        )
        self.img_path_column = img_path_column
        self.channel_column = channel_column
        self.scene_column = scene_column
        self.resolution_column = resolution_column
        self.time_start_column = time_start_column
        self.time_stop_column = time_stop_column
        self.time_step_column = time_step_column
        self.extra_columns = list(extra_columns)
        if spatial_dims not in (2, 3):
            raise ValueError(f"`spatial_dims` must be 2 or 3, got {spatial_dims}")
        self.spatial_dims = spatial_dims
        data = self.get_per_file_args(df)

        super().__init__(data, transform, **cache_kwargs)

    def _get_scenes(self, row, img):
        scenes = row.get(self.scene_column, -1)
        if scenes != -1:
            scenes = scenes.strip().split(",")
            for scene in scenes:
                if scene not in img.scenes:
                    raise ValueError(
                        f"For image {row[self.img_path_column]} unable to find scene `{scene}`, available scenes are {img.scenes}"
                    )
        else:
            scenes = img.scenes
        return scenes

    def _get_timepoints(self, row, img):
        start = row.get(self.time_start_column, 0)
        stop = row.get(self.time_stop_column, -1)
        step = row.get(self.time_step_column, 1)
        timepoints = range(start, stop + 1, step) if stop > 0 else range(img.dims.T)
        return list(timepoints)

    def get_per_file_args(self, df):
        img_data = []
        for row in tqdm.tqdm(df.itertuples()):
            row_data = []
            row = row._asdict()
            img = BioImage(row[self.img_path_column])
            scenes = self._get_scenes(row, img)
            for scene in scenes:
                img.set_scene(scene)
                timepoints = self._get_timepoints(row, img)
                for timepoint in timepoints:
                    image_loading_args = {
                        "dimension_order_out": "C" + "ZYX"[-self.spatial_dims :],
                        "C": row[self.channel_column],
                        "scene": scene,
                        "T": timepoint,
                        "original_path": row[self.img_path_column],
                        "resolution": row.get(self.resolution_column, 0),
                    }
                    extra_columns = {col: row.get(col) for col in self.extra_columns}
                    extra_columns.update(image_loading_args)
                    row_data.append(extra_columns)
            img_data.extend(row_data)
        return img_data
