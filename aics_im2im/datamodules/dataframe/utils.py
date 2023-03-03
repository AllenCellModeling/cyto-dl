import re
from itertools import chain
from typing import Iterator, List

import numpy as np

try:
    import modin.pandas as pd
except ModuleNotFoundError:
    import pandas as pd

from random import shuffle as random_shuffle

from monai.data import Dataset, PersistentDataset
from monai.transforms import Compose
from omegaconf import DictConfig, ListConfig
from torch.utils.data import BatchSampler, RandomSampler
from upath import UPath as Path

from aics_im2im.dataframe import read_dataframe


# randomly switch between generating batches from each sampler
class AlternatingBatchSampler(BatchSampler):
    def __init__(self, subset, head_allocation_column, batch_size, drop_last=False, shuffle=True):
        super().__init__(None, batch_size, drop_last)
        if head_allocation_column is None:
            raise ValueError("head_allocation_column must be defined if using batch sampler.")

        # order is   subset.monai_dataset.dataframewrapper.dataframe
        head_names = subset.dataset.data.df[head_allocation_column].unique()

        # pytorch subsets consist of the original dataset plus a list of `subset_indices`. when provided
        # an index `i` in getitem, subsets return `subset_indices[i]`. Since we are indexing into the
        # subset indices instead of the indices of the original dataframe, we have to reset the index
        # of the subsetted dataframe
        subset_df = subset.dataset.data.df.iloc[subset.indices].reset_index()
        samplers = []
        for name in head_names:
            # returns an index into dataset.indices where head == name
            head_indices = subset_df.index[subset_df[head_allocation_column] == name].to_list()
            if len(head_indices) == 0:
                raise ValueError(
                    f"Dataset must contain examples of head {name}. Please increase the value of subsample."
                )
            samplers.append(RandomSampler(head_indices))

        self.samplers = samplers
        self.shuffle = shuffle
        self.sampler_generator = self._sampler_generator()

    def _sampler_generator(self):
        # for now include equal numbers of all samplers
        steps_per_epoch = self.__len__()
        sampler_order = [[i] * steps_per_epoch for i in range(len(self.samplers))]
        # sampler0, sampler1, ..., sampler n, sampler 0, sampler1, ..., sampler n...
        interleaved_sampler_order = [
            _ for sampler_tuple in zip(*sampler_order) for _ in sampler_tuple
        ]
        if self.shuffle:
            random_shuffle(interleaved_sampler_order)

        for sampler_idx in range(len(interleaved_sampler_order)):
            yield self.samplers[sampler_idx]

    def __iter__(self) -> Iterator[List[int]]:
        try:
            sampler = next(self.sampler_generator)
        except StopIteration:
            self.sampler_generator = self._sampler_generator()
            sampler = next(self.sampler_generator)

        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:
            sampler_iter = iter(sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        sampler_len = np.min([len(sampler) for sampler in self.samplers])
        if self.drop_last:
            return sampler_len // self.batch_size  # type: ignore[arg-type]
        else:
            return (sampler_len + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]


def get_canonical_split_name(split):
    for canon in ("train", "val", "test", "predict"):
        if split.startswith(canon) or canon.startswith(split):
            return canon
    raise ValueError


def get_dataset(dataframe, transform, split, cache_dir=None):
    data = _DataframeWrapper(dataframe)
    if cache_dir is not None and split in ("train", "val"):
        return PersistentDataset(data, transform=transform, cache_dir=cache_dir)
    return Dataset(data, transform=transform)


def make_single_dataframe_splits(
    dataframe_path,
    transforms,
    split_column,
    columns=None,
    just_inference=False,
    split_map=None,
    cache_dir=None,
):
    dataframe = read_dataframe(dataframe_path, columns)
    dataframe[split_column] = dataframe[split_column].astype(np.dtype("O"))

    if not just_inference:
        assert dataframe.dtypes[split_column] == np.dtype("O")

    if split_map is not None:
        dataframe[split_column] = dataframe[split_column].replace(split_map)

    split_names = dataframe[split_column].unique().tolist()
    if not just_inference:
        assert set(split_names).issubset(
            {"train", "training", "valid", "val", "validation", "test", "testing"}
        )

    if split_column != "split":
        dataframe["split"] = dataframe[split_column].apply(get_canonical_split_name)

    datasets = {}
    if not just_inference:
        for split in ("train", "val", "test"):
            if cache_dir is not None:
                _split_cache = Path(cache_dir) / split
                _split_cache.mkdir(exist_ok=True, parents=True)
            else:
                _split_cache = None
            datasets[split] = get_dataset(
                dataframe.loc[dataframe["split"].str.startswith(split)],
                transforms[split],
                split,
                _split_cache,
            )

    datasets["predict"] = get_dataset(
        dataframe, transform=transforms["predict"], split="predict", cache_dir=cache_dir
    )
    return datasets


def make_multiple_dataframe_splits(
    split_path, transforms, columns=None, just_inference=False, cache_dir=None
):
    split_path = Path(split_path)
    datasets = {}
    predict_df = []

    for fpath in chain(split_path.glob("*.csv"), split_path.glob("*.parquet")):
        split = re.findall(r"(.*)\.(?:csv|parquet)", fpath.name)[0]
        split = get_canonical_split_name(split)
        dataframe = read_dataframe(fpath, required_columns=columns)
        dataframe["split"] = split

        if cache_dir is not None:
            _split_cache = Path(cache_dir) / split
            _split_cache.mkdir(exist_ok=True, parents=True)
        else:
            _split_cache = None

        if not just_inference:
            datasets[split] = get_dataset(dataframe, transforms[split], split, _split_cache)
        predict_df.append(dataframe.copy())

    predict_df = pd.concat(predict_df)
    datasets["predict"] = get_dataset(
        predict_df,
        transform=transforms["predict"],
        split="predict",
        cache_dir=cache_dir,
    )

    return datasets


def parse_transforms(transforms):
    if not isinstance(transforms, (DictConfig, dict)):
        transforms = {split: transforms for split in ["train", "val", "test", "predict"]}

    for k, v in transforms.items():
        if isinstance(v, (list, tuple, ListConfig)):
            v = Compose(v)
            transforms[k] = v

        transforms[get_canonical_split_name(k)] = v

    for k in transforms:
        if isinstance(transforms[k], str):
            assert transforms[k] in transforms
            transforms[k] = transforms[transforms[k]]

    for split in ("train", "val", "test"):
        if split not in transforms:
            raise ValueError(f"'{split}' missing from transforms dict.")

    if "predict" not in transforms:
        transforms["predict"] = transforms["test"]

    return transforms


class _DataframeWrapper:
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        return self.df.iloc[ix].to_dict()
