from pathlib import Path
from typing import Optional, Union

from monai.data import DataLoader
from monai.transforms import Compose

from .smartcache import SmartcacheDatamodule
from .utils import AlternatingBatchSampler


class GroupedSmartcacheDatamodule(SmartcacheDatamodule):
    """Datamodule for large CZI datasets that don't fit in memory."""

    def __init__(
        self,
        csv_path: Union[Path, str],
        transforms: Compose = None,
        img_data: Optional[Union[Path, str]] = None,
        n_val: int = 20,
        pct_val: float = 0.1,
        img_path_column: str = "raw",
        channel_column: str = "ch",
        spatial_dims: int = 3,
        neighboring_timepoints: bool = False,
        num_workers: int = 4,
        cache_rate: float = 0.5,
        target_columns: str = None,
        grouping_column: str = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        csv_path: Union[Path, str]
            path to csv with image in `img_path_column` and channel in `channel_column`
        transforms: Compose
            Monai transforms to apply to each image. Should start with a transform that uses AICSimageio for image reading
        img_data: Union[Path, str]
            csv_path generated by get_per_file_args that enumerates scenes and timepoints for each image in csv_path
        n_val: int
            number of validation images to use. Minimum of pct_val * n_images and n_val is used.
        pct_val: float
            percentage of images to use for validation. Minimum of pct_val * n_images and n_val is used.
        img_path_column: str
            column in csv_path that contains the path to the image
        channel_column: str
            column in csv_path that contains the channel to use
        spatial_dims: int
            number of spatial dimensions in the image
        neighboring_timepoints: bool
            whether to return T and T+1 as a 2 channel image, useful for models incorporating time
        num_workers: int
            number of workers to use for loading data. Most be specified here to schedule replacement workers for cache data
        cache_rate: float
            percentage of data to cache
        target_columns: str = None
            column names in csv corresponding to ground truth types to alternate between
            during training
        grouping_column: str = None
            column names in csv corresponding to a factor that should be homogeneous across
            a batch
        """
        # make sure dataloader kwargs doesn't contain invalid arguments
        kwargs.pop("drop_last", None)
        kwargs.pop("batch_sampler", None)
        kwargs.pop("sampler", None)
        super().__init__(
            csv_path=csv_path,
            transforms=transforms,
            img_data=img_data,
            n_val=n_val,
            pct_val=pct_val,
            img_path_column=img_path_column,
            channel_column=channel_column,
            spatial_dims=spatial_dims,
            neighboring_timepoints=neighboring_timepoints,
            num_workers=num_workers,
            cache_rate=cache_rate,
        )
        self.group_column = grouping_column
        self.target_columns = target_columns

    def make_dataloader(self, split):
        # smartcachedataset can't have persistent workers
        self.kwargs["persistent_workers"] = split not in ("train", "val")
        if "num_workers" in self.kwargs:
            del self.kwargs["num_workers"]
        self.kwargs["shuffle"] = self.kwargs.get("shuffle", False) and split == "train"

        subset = self.datasets[split]

        batch_sampler = AlternatingBatchSampler(
            subset,
            batch_size=self.kwargs.pop("batch_size"),
            drop_last=True,
            shuffle=self.kwargs.pop("shuffle"),
            target_columns=self.target_columns,
            grouping_column=self.group_column,
        )

        return DataLoader(
            self.datasets[split],
            num_workers=self.num_workers,
            batch_sampler=batch_sampler**self.kwargs,
        )