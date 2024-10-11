from typing import Dict, Optional, Sequence, Union

from monai.data import DataLoader
from upath import UPath as Path

from .dataframe_datamodule import DataframeDatamodule
from .utils import AlternatingBatchSampler


class GroupedDataframeDatamodule(DataframeDatamodule):
    """A DataframeDatamodule modified for cases where batches should be grouped by some criterion
    leveraging an AlternatingBatchSampler.

    The two use cases currently supported are 1. multitask training where ground truths are only
    available for one task at a time and 2. training where batches are grouped by some
    characteristic of the images.
    """

    def __init__(
        self,
        path: Union[Path, str],
        transforms: Dict,
        split_column: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
        split_map: Optional[Dict] = None,
        just_inference: bool = False,
        cache_dir: Optional[Union[Path, str]] = None,
        subsample: Optional[Dict] = None,
        refresh_subsample: bool = False,
        seed: int = 42,
        smartcache_args: Optional[Dict] = None,
        target_columns: str = None,
        grouping_column: str = None,
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

        subsample: Optional[Dict] = None
            Dictionary with a key per split ("train", "val", "test"), and the
            number of samples of each split to use per epoch. If `None` (default),
            use all the samples in each split per epoch.

        refresh_subsample: bool = False
            Whether to refresh subsample each time dataloader is called

        seed: int = 42
            random seed

        smartcache_args: Optional[Dict] = None
            Arguments to pass to SmartcacheDataset

        target_columns: str = None
            column names in csv corresponding to ground truth types to alternate between
            during training

        grouping_column: str = None
            column names in csv corresponding to a factor that should be homogeneous across
            a batch

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
            subsample=subsample,
            refresh_subsample=refresh_subsample,
            seed=seed,
            smartcache_args=smartcache_args,
            **dataloader_kwargs,
        )
        self.group_column = grouping_column
        self.target_columns = target_columns

    def make_dataloader(self, split):
        kwargs = {**self.dataloader_kwargs}
        kwargs["shuffle"] = kwargs.get("shuffle", True) and split == "train"
        subset = self.get_dataset(split)

        batch_sampler = AlternatingBatchSampler(
            subset,
            batch_size=kwargs.pop("batch_size"),
            drop_last=True,
            shuffle=kwargs.pop("shuffle"),
            target_columns=self.target_columns,
            grouping_column=self.group_column,
        )
        return DataLoader(dataset=subset, batch_sampler=batch_sampler, **kwargs)
