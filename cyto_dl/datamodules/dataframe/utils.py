import re
from itertools import chain, repeat
from typing import Iterator, List

import numpy as np
import pyarrow as pa

try:
    import modin.pandas as pd
except ModuleNotFoundError:
    import pandas as pd

import random
from typing import Sequence

from monai.data import Dataset, PersistentDataset, SmartCacheDataset
from monai.transforms import Compose, Transform
from omegaconf import DictConfig, ListConfig
from torch.utils.data import BatchSampler, Sampler, Subset, SubsetRandomSampler
from upath import UPath as Path

from cyto_dl.dataframe import read_dataframe


class RemoveNaNKeysd(Transform):
    """Transform to remove 'nan' keys from data dictionary.

    When combined with adding `allow_missing_keys=True` to transforms and the alternating batch
    sampler, this allows multi-task training when only one target is available at a time.
    """

    def __init__(img):
        super().__init__()

    def __call__(self, img):
        new_data = {k: v for k, v in img.items() if not pd.isna(v)}
        return new_data


# randomly switch between generating batches from each sampler
class AlternatingBatchSampler(BatchSampler):
    """Subclass of pytorch's `BatchSampler` that alternates between sampling from mutually
    exclusive columns of a dataframe dataset."""

    def __init__(
        self,
        subset: Subset,
        target_columns: Sequence[str] = None,
        grouping_column: Sequence[str] = None,
        batch_size: int = 1,
        drop_last: bool = False,
        shuffle: bool = False,
        sampler: Sampler = SubsetRandomSampler,
    ):
        """
        Parameters
        ----------
        subset: Subset
            Subset of monai dataset wrapping a dataframe
        target_columns: Sequence[str]
            names of columns in `subset` dataframe representing types of ground truth images to alternate between
        batch_size:int
            Size of batch
        drop_last:bool=False
            Whether to drop last incomplete batch
        shuffle:bool=False
            Whether to randomly select between columns in `target_columns`. If False, batches will follow the order of
            `target_columns`
        sampler:Sampler=SubsetRandomSampler
            Sampler to sample from each column in `target_columns`
        """
        assert (
            grouping_column is not None or target_columns is not None
        ), "Either target column or grouping column must be provided"
        super().__init__(None, batch_size, drop_last)
        # pytorch subsets consist of the original dataset plus a list of `subset_indices`. when provided
        # an index `i` in getitem, subsets return `subset_indices[i]`. Since we are indexing into the
        # subset indices instead of the indices of the original dataframe, we have to reset the index
        # of the subsetted dataframe

        # order is subset.monai_dataset.dataframewrapper.dataframe
        idx_to_take = list(subset.indices)
        subset_df = subset.dataset.data.dataframe.take(idx_to_take).to_pandas().reset_index()
        samplers = []
        if target_columns is not None:
            for name in target_columns:
                # returns an index into dataset.indices where head column is not empty
                head_indices = subset_df.index[~subset_df[name].isna()].values
                if len(head_indices) == 0:
                    raise ValueError(
                        f"Dataset must contain examples of head {name}. Please increase the value of subsample."
                    )
                samplers.append(sampler(head_indices))
        else:
            grouping_options = set(
                subset.dataset.data.dataframe[grouping_column].unique().to_pylist()
            )

            seen_keys = set()
            for group_key, group in subset_df.groupby(grouping_column):
                samplers.append(sampler(group.index.values))
                seen_keys.add(group_key)

            unseen_keys = grouping_options - seen_keys
            if unseen_keys:
                raise ValueError(
                    f"Dataset must contain examples of groups {unseen_keys}."
                    "Please increase the value of subsample."
                )

        self.samplers = samplers
        self.shuffle = shuffle
        self._sampler_generator()

    def _sampler_generator(self):
        self.sampler_iterators = [iter(s) for s in self.samplers]

        # for now include equal numbers of all samplers
        # worst case shuffle, we get [0,0,0,0,...., 1, 1, 1, 1]. after the 0s are done, stop iteration
        # is raised but we're only halfway through the epoch! this prevents that by never fully exhausting
        # the samplers
        samples_per_sampler = len(self) // len(self.samplers)

        if samples_per_sampler <= 0:
            raise ValueError("Insufficient examples per task head. Please decrease batch size.")

        interleaved_sampler_order = repeat(range(len(self.samplers)), samples_per_sampler)
        interleaved_sampler_order = chain.from_iterable(interleaved_sampler_order)
        interleaved_sampler_order = list(interleaved_sampler_order)

        if self.shuffle:
            random.shuffle(interleaved_sampler_order)

        self.sampler_order = interleaved_sampler_order

    def __iter__(self) -> Iterator[List[int]]:
        for sampler_ix in self.sampler_order:
            try:
                yield [next(self.sampler_iterators[sampler_ix]) for _ in range(self.batch_size)]
            except StopIteration:
                self._sampler_generator()
                yield [next(self.sampler_iterators[sampler_ix]) for _ in range(self.batch_size)]

    def __len__(self) -> int:
        min_num_samples = min(len(sampler) for sampler in self.samplers)
        min_num_batches = min_num_samples // self.batch_size
        return min_num_batches * len(self.samplers)


def get_canonical_split_name(split):
    for canon in ("train", "val", "test", "predict"):
        if split.startswith(canon) or canon.startswith(split):
            return canon
    return None


def get_dataset(dataframe, transform, split, cache_dir=None, smartcache_args=None):
    data = _DataframeWrapper(dataframe)
    if smartcache_args is not None and split in ("train", "val"):
        cache_rate = smartcache_args.get("cache_rate", 0.1)
        if cache_rate * len(data) >= 1:
            print(f"Initializing SmartCacheDataset for {split}")
            return SmartCacheDataset(
                data,
                transform=transform,
                cache_rate=cache_rate,
                replace_rate=smartcache_args.get("replace_rate", 0.1),
                num_init_workers=smartcache_args.get("num_init_workers", 4),
                num_replace_workers=smartcache_args.get("num_replace_workers", 4),
            )
        else:
            print(
                f"Cache rate {cache_rate} too low for {split} dataset with size {len(data)}, using Dataset."
            )
    if cache_dir is not None and split in ("train", "val"):
        print(f"Initializing PersistentDataset for {split} at {cache_dir}")
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
    smartcache_args=None,
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
                smartcache_args=smartcache_args,
            )

    datasets["predict"] = get_dataset(
        dataframe,
        transform=transforms["predict"],
        split="predict",
        cache_dir=cache_dir,
        smartcache_args=smartcache_args,
    )
    return datasets


def make_multiple_dataframe_splits(
    split_path,
    transforms,
    columns=None,
    just_inference=False,
    cache_dir=None,
    smartcache_args=None,
):
    split_path = Path(split_path)
    datasets = {}
    predict_df = []

    for fpath in chain(split_path.glob("*.csv"), split_path.glob("*.parquet")):
        split = re.findall(r"(.*)\.(?:csv|parquet)", fpath.name)[0]
        split = get_canonical_split_name(split)
        if split is None:
            print(f"Skipping {fpath} as it does not match any known split name.")
            continue
        dataframe = read_dataframe(fpath, required_columns=columns)
        dataframe["split"] = split

        if cache_dir is not None:
            _split_cache = Path(cache_dir) / split
            _split_cache.mkdir(exist_ok=True, parents=True)
        else:
            _split_cache = None

        if not just_inference:
            datasets[split] = get_dataset(
                dataframe, transforms[split], split, _split_cache, smartcache_args=smartcache_args
            )
        predict_df.append(dataframe.copy())

    if len(predict_df) == 0:
        raise ValueError(f"No valid dataframe files found in {split_path}")

    predict_df = pd.concat(predict_df)
    datasets["predict"] = get_dataset(
        predict_df,
        transform=transforms["predict"],
        split="predict",
        cache_dir=cache_dir,
        smartcache_args=smartcache_args,
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
    """Class to wrap a pandas DataFrame in a pytorch Dataset. In practice, at AICS we use this to
    wrap manifest dataframes that point to the image files that correspond to a cell. The `loaders`
    dict contains a loading function for each key, normally consisting of a function to load the
    contents of a file from a path.

    Parameters
    ----------
    dataframe: pd.DataFrame
        The file which points to or contains the data to be loaded
    """

    def __init__(self, dataframe):
        self.dataframe = pa.Table.from_pandas(dataframe)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return {k: v.pop() for k, v in self.dataframe.take([idx]).to_pydict().items()}
