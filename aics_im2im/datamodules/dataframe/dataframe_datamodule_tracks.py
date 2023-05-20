from typing import Dict, Optional, Sequence, Union

from monai.data import DataLoader
from upath import UPath as Path

from .dataframe_datamodule import DataframeDatamodule
from .track_sampler import TrackSampler


def get_track_indices(gr, time_col):
    return gr.sort_values(by=time_col, ascending=True).index.tolist()


class DataframeDatamoduleTracks(DataframeDatamodule):
    """A DataframeDatamodule modified for multi-task settings, leveraging an
    AlternatingBatchSampler."""

    def __init__(
        self,
        path: Union[Path, str],
        transforms: Dict,
        split_column: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
        split_map: Optional[Dict] = None,
        just_inference: bool = False,
        cache_dir: Optional[Union[Path, str]] = None,
        seed: int = 42,
        window_overlap: Optional[int] = 0,
        track_id_col: str = "track_id",
        time_col: str = "T",
        **dataloader_kwargs,
    ):
        """
        Parameters
        ----------
        path: Union[Path, str]
            Path to a dataframe file

        transforms: Dict
            Transforms specifications for each given split.

        split_column: Optional[str] = None
            Name of a column in the dataset which can be used to create train, val, test
            splits.

        columns: Optional[Sequence[str]] = None
            List of columns to load from the dataset, in case it's a parquet file.
            If None, load everything.

        split_map: Optional[Dict] = None
            TODO: document this argument

        just_inference: bool = False
            Whether this datamodule will be used for just inference
            (testing/prediction).
            If so, the splits are ignored and the whole dataset is used.

        cache_dir: Optional[Union[Path, str]] = None
            Path to a directory in which to store cached transformed inputs, to
            accelerate batch loading.

        seed: int = 42
            random seed

        dataloader_kwargs:
            Additional keyword arguments are passed to the
            torch.utils.data.DataLoader class when instantiating it (aside from
            `shuffle` which is only used for the train dataloader).
            Among these args are `num_workers`, `batch_size`, `shuffle`, etc.
            See the PyTorch docs for more info on these args:
            https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        """

        # make sure dataloader kwargs doesn't contain invalid arguments
        dataloader_kwargs.pop("drop_last", None)
        dataloader_kwargs.pop("batch_sampler", None)
        dataloader_kwargs.pop("sampler", None)

        super().__init__(
            path=path,
            transforms=transforms,
            split_column=split_column,
            columns=columns,
            split_map=split_map,
            just_inference=just_inference,
            cache_dir=cache_dir,
            seed=seed,
            **dataloader_kwargs,
        )

        self.window_overlap = window_overlap
        self.track_id_col = track_id_col
        self.time_col = time_col

    def make_dataloader(self, split):
        kwargs = dict(**self.dataloader_kwargs)
        kwargs["shuffle"] = kwargs.get("shuffle", True) and split == "train"

        # we're always getting the full dataset here,
        subset = self.get_dataset(split)

        track_indices = subset.dataset.data.df.groupby(  # access the underlying dataframe
            self.track_id_col
        ).apply(get_track_indices)

        batch_sampler = TrackSampler(
            track_indices, batch_size=kwargs.pop("batch_size"), overlap=self.window_overlap
        )
        return DataLoader(subset, batch_sampler=batch_sampler, **kwargs)
