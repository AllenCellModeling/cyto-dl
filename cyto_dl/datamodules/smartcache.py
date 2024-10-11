from itertools import chain
from pathlib import Path
from typing import Optional, Union

import dask
import numpy as np
import pandas as pd
from bioio import BioImage
from dask.diagnostics import ProgressBar
from lightning import LightningDataModule
from monai.data import DataLoader
from monai.data.dataset import CacheDataset, Dataset, SmartCacheDataset
from monai.transforms import Compose
from sklearn.model_selection import train_test_split


class SmartcacheDatamodule(LightningDataModule):
    """Datamodule for large CZI datasets that don't fit in memory."""

    def __init__(
        self,
        csv_path: Optional[Union[Path, str]] = None,
        transforms: Compose = None,
        img_data: Optional[Union[Path, str]] = None,
        n_val: int = 20,
        pct_val: float = 0.1,
        img_path_column: str = "raw",
        channel_column: str = "ch",
        spatial_dims: int = 3,
        num_neighbors: int = 0,
        num_workers: int = 4,
        cache_rate: float = 0.5,
        replace_rate: float = 0.1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        csv_path: Union[Path, str]
            path to csv with image in `img_path_column` and channel in `channel_column`
        transforms: Compose
            Monai transforms to apply to each image. Should start with a transform that uses bioio for image reading
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
        num_neighbors: int
            number of neighboring timepoints to use
        num_workers: int
            number of workers to use for loading data. Most be specified here to schedule replacement workers for cache data
        cache_rate: float
            percentage of data to cache
        replace_rate: float
            percentage of data to replace
        kwargs:
            additional arguments to pass to DataLoader
        """
        super().__init__()
        self.img_data = {}
        if isinstance(img_data, (str, Path)):
            # read img_data if it's a path, otherwise set to empty dict
            self.img_data["train"] = [
                row._asdict()
                for row in pd.read_csv(Path(img_data) / "train_img_data.csv").itertuples()
            ]
            self.img_data["val"] = [
                row._asdict()
                for row in pd.read_csv(Path(img_data) / "val_img_data.csv").itertuples()
            ]
        elif csv_path is not None:
            self.csv_path = Path(csv_path)
            (self.csv_path.parents[0] / "loaded_data").mkdir(exist_ok=True, parents=True)
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("csv_path or img_data must be specified")
        self.num_workers = num_workers
        self.kwargs = kwargs

        self.n_val = n_val
        self.pct_val = pct_val

        self.datasets = {}
        self.img_path_column = img_path_column
        self.channel_column = channel_column
        self.spatial_dims = spatial_dims
        self.transforms = transforms
        self.num_neighbors = num_neighbors
        self.cache_rate = cache_rate
        self.replace_rate = replace_rate

    def _get_scenes(self, img):
        """Get the number of scenes in an image."""
        return img.scenes

    def _get_timepoints(self, img):
        """Get the number of timepoints in an image."""
        timepoints = list(range(img.dims.T))
        if self.num_neighbors > 0:
            return timepoints[: -self.num_neighbors]
        return timepoints

    @dask.delayed
    def _get_file_args(self, row):
        row = row._asdict()
        img = BioImage(row[self.img_path_column])
        scenes = self._get_scenes(img)
        timepoints = self._get_timepoints(img)
        img_data = []
        use_neighbors = self.num_neighbors > 0
        for scene in scenes:
            for timepoint in timepoints:
                img_data.append(
                    {
                        "dimension_order_out": (
                            "ZYX"[-self.spatial_dims :]
                            if not use_neighbors
                            else "T" + "ZYX"[-self.spatial_dims :]
                        ),
                        "C": row[self.channel_column],
                        "scene": scene,
                        "T": (
                            timepoint
                            if not use_neighbors
                            else [timepoint + i for i in range(self.num_neighbors + 1)]
                        ),
                        "original_path": row[self.img_path_column],
                    }
                )
        return img_data

    def get_per_file_args(self, df):
        """Parallelize getting the image loading arguments enumerating all
        timepoints/channels/scenes for each file in the dataframe."""
        with ProgressBar():
            img_data = dask.compute(*[self._get_file_args(row) for row in df.itertuples()])
        img_data = list(chain.from_iterable(img_data))
        return img_data

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit":
            if "train" not in self.img_data or "val" not in self.img_data:
                # update img_data
                image_data = self.get_per_file_args(self.df)
                val_size = np.min([self.n_val, int(len(image_data) * self.pct_val)])
                val_size = np.max([val_size, 1])
                self.img_data["train"], self.img_data["val"] = train_test_split(
                    image_data, test_size=val_size
                )

                print("Train images:", len(self.img_data["train"]))
                print("Val images:", len(self.img_data["val"]))

                pd.DataFrame(self.img_data["train"]).to_csv(
                    f"{self.csv_path.parents[0]}/loaded_data/train_img_data.csv",
                    index=False,
                )
                pd.DataFrame(self.img_data["val"]).to_csv(
                    f"{self.csv_path.parents[0]}/loaded_data/val_img_data.csv", index=False
                )

            self.datasets["train"] = SmartCacheDataset(
                self.img_data["train"],
                transform=self.transforms["train"],
                cache_rate=self.cache_rate,
                num_replace_workers=self.num_workers,
                num_init_workers=self.num_workers,
                replace_rate=self.replace_rate,
            )
            self.datasets["val"] = CacheDataset(
                self.img_data["val"],
                transform=self.transforms["valid"],
                cache_rate=1.0,
                num_workers=self.num_workers,
            )

        elif stage in ("test", "predict"):
            self.img_data[stage] = self.get_per_file_args(self.df)
            self.datasets[stage] = Dataset(self.img_data[stage], transform=self.transforms[stage])

    def make_dataloader(self, split):
        # smartcachedataset can't have persistent workers
        self.kwargs["persistent_workers"] = split not in ("train", "val")
        if "num_workers" in self.kwargs:
            del self.kwargs["num_workers"]
        return DataLoader(
            self.datasets[split],
            num_workers=self.num_workers,
            **self.kwargs,
        )

    def train_dataloader(self):
        return self.make_dataloader("train")

    def val_dataloader(self):
        return self.make_dataloader("val")

    def test_dataloader(self):
        return self.make_dataloader("test")

    def predict_dataloader(self):
        return self.make_dataloader("predict")
