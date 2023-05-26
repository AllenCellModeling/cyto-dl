from typing import Dict, Optional, Union

from upath import UPath as Path

from .czi import make_CZI_dataloader
from .dataframe.dataframe_datamodule import DataframeDatamodule
from .dataframe.utils import parse_transforms


class CZIDatamodule(DataframeDatamodule):
    def __init__(
        self,
        path: Union[Path, str],
        transforms: Dict,
        split_column: Optional[str] = None,
        img_path_column: Optional[str] = None,
        channel_column: Optional[str] = None,
        out_key: Optional[str] = "img",
        scene_column: Optional[str] = "scene",
        time_start_column: Optional[str] = None,
        time_stop_column: Optional[str] = None,
        time_step_column: Optional[str] = None,
        subsample: Optional[Dict] = None,
        refresh_subsample: bool = False,
        seed: int = 42,
        spatial_dims: int = 3,
        target_columns: str = None,
        grouping_column: str = None,
        **dataloader_kwargs,
    ):
        dataloader_kwargs.pop("drop_last", None)
        dataloader_kwargs.pop("batch_sampler", None)
        dataloader_kwargs.pop("sampler", None)

        columns = [
            col
            for col in (
                split_column,
                img_path_column,
                channel_column,
                scene_column,
                time_start_column,
                time_step_column,
                time_stop_column,
                target_columns,
                grouping_column,
            )
            if col is not None
        ]

        super().__init__(
            path=path,
            transforms=None,
            split_column=split_column,
            columns=columns,
            subsample=subsample,
            refresh_subsample=refresh_subsample,
            seed=seed,
            **dataloader_kwargs,
        )
        self.transforms = parse_transforms(transforms)
        self.grouping_column = grouping_column
        self.target_columns = target_columns
        self.img_path_column = img_path_column
        self.channel_column = channel_column
        self.out_key = out_key
        self.scene_column = scene_column
        self.time_start_column = time_start_column
        self.time_stop_column = time_stop_column
        self.time_step_column = time_step_column
        self.spatial_dims = spatial_dims

    def make_dataloader(self, split):
        kwargs = dict(**self.dataloader_kwargs)
        kwargs["shuffle"] = kwargs.get("shuffle", True) and split == "train"
        subset = self.get_dataset(split)
        return make_CZI_dataloader(
            img_path_column=self.img_path_column,
            channel_column=self.channel_column,
            out_key=self.out_key,
            df=subset.dataset.data.df.iloc[subset.indices].reset_index(),
            spatial_dims=self.spatial_dims,
            scene_column=self.scene_column,
            time_start_column=self.time_start_column,
            time_stop_column=self.time_stop_column,
            time_step_column=self.time_step_column,
            transforms=self.transforms[split],
            grouped=self.grouping_column is not None,
            target_columns=self.target_columns,
            grouping_column=self.grouping_column,
            **kwargs,
        )
