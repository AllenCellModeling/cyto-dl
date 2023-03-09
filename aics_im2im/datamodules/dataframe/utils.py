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
from torch.utils.data import BatchSampler, SubsetRandomSampler
from upath import UPath as Path

from aics_im2im.dataframe import read_dataframe

from monai.transforms import Transform

class RemoveNaNKeysd(Transform):
    ''''
    Transform to remove 'nan' keys from data dictionary. When combined with adding
    `allow_missing_keys=True` to transforms and the alternating batch sampler, this
    allows multi-task training when only one target is available at a time.
    '''
    def __init__(img):
        super().__init__()
    def __call__(self, img):
        new_data = {k: v for k, v in img.items() if not pd.isna(v)}
        return new_data


# randomly switch between generating batches from each sampler
class AlternatingBatchSampler(BatchSampler):
    def __init__(
        self,
        subset,
        target_columns,
        batch_size,
        drop_last=False,
        shuffle=True,
        sampler=SubsetRandomSampler,
    ):
        super().__init__(None, batch_size, drop_last)
        if target_columns is None:
            raise ValueError("target_columns must be defined if using batch sampler.")
        # pytorch subsets consist of the original dataset plus a list of `subset_indices`. when provided
        # an index `i` in getitem, subsets return `subset_indices[i]`. Since we are indexing into the
        # subset indices instead of the indices of the original dataframe, we have to reset the index
        # of the subsetted dataframe

        # order is   subset.monai_dataset.dataframewrapper.dataframe
        subset_df = subset.dataset.data.df.iloc[subset.indices].reset_index()
        samplers = []
        for name in target_columns:
            # returns an index into dataset.indices where head column is not emtpy
            head_indices = subset_df.index[subset_df[name].isna()].to_list()
            if len(head_indices) == 0:
                raise ValueError(
                    f"Dataset must contain examples of head {name}. Please increase the value of subsample."
                )
            samplers.append(sampler(head_indices))

        self.samplers = samplers

        self.shuffle = shuffle
        self.sampler_generator = self._sampler_generator()

    def _sampler_generator(self):
        self.sampler_iterators = [iter(s) for s in self.samplers]
        # for now include equal numbers of all samplers
        samples_per_sampler = self.__len__() // len(self.samplers)
        if samples_per_sampler == 0:
            raise ValueError("Insufficient examples per task head. Please decrease batch size.")
        sampler_order = [[i] * samples_per_sampler for i in range(len(self.samplers))]
        # sampler0, sampler1, ..., sampler n, sampler 0, sampler1, ..., sampler n...
        interleaved_sampler_order = [
            _ for sampler_tuple in zip(*sampler_order) for _ in sampler_tuple
        ]

        if self.shuffle:
            random_shuffle(interleaved_sampler_order)
        self.sampler_order = interleaved_sampler_order

    def get_next_sampler(self):
        if len(self.sampler_order) > 0:
            return self.sampler_iterators[self.sampler_order.pop()]
        self._sampler_generator()
        raise StopIteration

    def __iter__(self) -> Iterator[List[int]]:
        sampler = self.get_next_sampler()
        while True:
            try:
                batch = [next(sampler) for _ in range(self.batch_size)]
                yield batch
                sampler = self.get_next_sampler()
            except StopIteration:
                break

    def __len__(self) -> int:
        sampler_len = np.min([len(sampler) for sampler in self.samplers]) * len(self.samplers)
        return sampler_len // self.batch_size  # type: ignore[arg-type]


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
