from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from bioio import BioImage
from monai.data import DataLoader, Dataset, MetaTensor
from monai.transforms import Compose, apply_transform
from omegaconf import ListConfig


class CZIDataset(Dataset):
    """Dataset converting a `.csv` file listing CZI files and some metadata into batches of single-
    scene, single-timepoint, single-channel images."""

    def __init__(
        self,
        csv_path: Union[Path, str],
        img_path_column: str,
        channel_column: str,
        out_key: str,
        spatial_dims: int = 3,
        scene_column: str = "scene",
        time_start_column: str = "start",
        time_stop_column: str = "stop",
        time_step_column: str = "step",
        transform: Optional[Callable] = None,
        dask_load: bool = True,
    ):
        """
        Parameters
        ----------
        csv_path: Union[Path, str]
            path to csv
        img_path_column: str
            column in `csv_path` that contains path to CZI file
        channel_column:str
            Column in `csv_path` that contains which channel to extract from CZI file. Should be an integer.
        out_key:str
            Key where single-scene/timepoint/channel is saved in output dictionary
        spatial_dims:int=3
            Spatial dimension of output image. Must be 2 for YX or 3 for ZYX
        scene_column:str="scene",
            Column in `csv_path` that contains scenes to extract from CZI file. If not specified, all scenes will
            be extracted. If multiple scenes are specified, they should be separated by a comma (e.g. `scene1,scene2`)
        time_start_column:str="start"
            Column in `csv_path` specifying which timepoint in timelapse CZI to start extracting. If any of `start_column`, `stop_column`, or `step_column`
            are not specified, all timepoints are extracted.
        time_stop_column:str="stop"
            Column in `csv_path` specifying which timepoint in timelapse CZI to stop extracting. If any of `start_column`, `stop_column`, or `step_column`
            are not specified, all timepoints are extracted.
        time_step_column:str="step"
            Column in `csv_path` specifying step between timepoints. For example, values in this column should be `2` if every other timepoint should be run.
            If any of `start_column`, `stop_column`, or `step_column` are not specified, all timepoints are extracted.
        transform: Optional[Callable] = None
            Callable to that accepts numpy array. For example, image normalization functions could be passed here.
        dask_load: bool = True
            Whether to use dask to load images. If False, full images are loaded into memory before extracting specified scenes/timepoints.
        """
        super().__init__(None, transform)
        df = pd.read_csv(csv_path)
        self.img_path_column = img_path_column
        self.channel_column = channel_column
        self.scene_column = scene_column
        self.time_start_column = time_start_column
        self.time_stop_column = time_stop_column
        self.time_step_column = time_step_column
        self.out_key = out_key
        if spatial_dims not in (2, 3):
            raise ValueError(f"`spatial_dims` must be 2 or 3, got {spatial_dims}")
        self.spatial_dims = spatial_dims
        self.dask_load = dask_load

        self.img_data = self.get_per_file_args(df)

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
        start = row.get(self.time_start_column, -1)
        stop = row.get(self.time_stop_column, -1)
        step = row.get(self.time_step_column, 1)
        timepoints = list(range(start, stop + 1, step))
        if np.any(np.array((start, stop, step)) == -1):
            timepoints = list(range(img.dims.T))
        return timepoints

    def get_per_file_args(self, df):
        img_data = []
        for row in df.itertuples():
            row = row._asdict()
            img = BioImage(row[self.img_path_column])
            scenes = self._get_scenes(row, img)
            timepoints = self._get_timepoints(row, img)
            for scene in scenes:
                for timepoint in timepoints:
                    img_data.append(
                        {
                            "img": img,
                            "dimension_order_out": "ZYX"[-self.spatial_dims :],
                            "C": row[self.channel_column],
                            "scene": scene,
                            "T": timepoint,
                            "original_path": row[self.img_path_column],
                        }
                    )
        return img_data

    def _metadata_to_str(self, metadata):
        return "_".join([] + [f"{k}={v}" for k, v in metadata.items()])

    def _ensure_channel_first(self, img):
        while len(img.shape) < self.spatial_dims + 1:
            img = np.expand_dims(img, 0)
        return img

    def _transform(self, index: int):
        img_data = self.img_data.pop()
        img = img_data.pop("img")
        original_path = img_data.pop("original_path")
        scene = img_data.pop("scene")
        img.set_scene(scene)
        if self.dask_load:
            data_i = img.get_image_dask_data(**img_data).compute()
        else:
            data_i = img.get_image_data(**img_data)
        img_data["scene"] = scene
        data_i = self._ensure_channel_first(data_i)
        output_img = (
            apply_transform(self.transform, data_i) if self.transform is not None else data_i
        )

        return {
            self.out_key: MetaTensor(
                output_img,
                meta={
                    "filename_or_obj": original_path.replace(".", self._metadata_to_str(img_data))
                },
            )
        }

    def __len__(self):
        return len(self.img_data)


def make_CZI_dataloader(
    csv_path,
    img_path_column,
    channel_column,
    out_key,
    spatial_dims=3,
    scene_column="scene",
    time_start_column="start",
    time_stop_column="stop",
    time_step_column="step",
    transforms=None,
    **dataloader_kwargs,
):
    """Function to create a CZI Dataset. Currently, this dataset is only useful during prediction
    and cannot be used for training or testing.

    Parameters
    ----------
    csv_path: Union[Path, str]
        path to csv
    img_path_column: str
        column in `csv_path` that contains path to CZI file
    channel_column:str
        Column in `csv_path` that contains which channel to extract from CZI file. Should be an integer.
    out_key:str
        Key where single-scene/timepoint/channel is saved in output dictionary
    spatial_dims:int=3
        Spatial dimension of output image. Must be 2 for YX or 3 for ZYX
    scene_column:str="scene",
        Column in `csv_path` that contains scenes to extract from CZI file. If not specified, all scenes will
        be extracted. If multiple scenes are specified, they should be separated by a comma (e.g. `scene1,scene2`)
    time_start_column:str="start"
        Column in `csv_path` specifying which timepoint in timelapse CZI to start extracting. If any of `start_column`, `stop_column`, or `step_column`
        are not specified, all timepoints are extracted.
    time_stop_column:str="stop"
        Column in `csv_path` specifying which timepoint in timelapse CZI to stop extracting. If any of `start_column`, `stop_column`, or `step_column`
        are not specified, all timepoints are extracted.
    time_step_column:str="step"
        Column in `csv_path` specifying step between timepoints. For example, values in this column should be `2` if every other timepoint should be run.
        If any of `start_column`, `stop_column`, or `step_column` are not specified, all timepoints are extracted.
    transform: Optional[Callable] = None
        Callable to that accepts numpy array. For example, image normalization functions could be passed here.
    """
    if isinstance(transforms, (list, tuple, ListConfig)):
        transforms = Compose(transforms)
    dataset = CZIDataset(
        csv_path,
        img_path_column,
        channel_column,
        out_key,
        spatial_dims,
        scene_column=scene_column,
        time_start_column=time_start_column,
        time_stop_column=time_stop_column,
        time_step_column=time_step_column,
        transform=transforms,
    )
    return DataLoader(dataset, **dataloader_kwargs)
