from collections import defaultdict

import monai
import numpy as np
import pyrootutils
import pytest
from torch.utils.data import SubsetRandomSampler

from cyto_dl.datamodules.dataframe.utils import (
    AlternatingBatchSampler,
    RemoveNaNKeysd,
    make_multiple_dataframe_splits,
)
from cyto_dl.image.io import MonaiBioReader

overrides = ["logger=[]", "++source_col=raw", "++target_col=seg"]


# using this because the sequential sampler returns range(len(indices)) instead of indices
class TestSampler:
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("sampler", [SubsetRandomSampler, TestSampler])
def test_alternating_batch_sampler(shuffle, sampler):
    root_dir = pyrootutils.find_root()
    transforms = monai.transforms.Compose(
        [
            RemoveNaNKeysd(),
            monai.transforms.LoadImaged(
                keys=["raw", "seg1", "seg2"],
                reader=MonaiBioReader(dimension_order_out="CZYX"),
                allow_missing_keys=True,
            ),
        ]
    )
    transform_dict = {key: transforms for key in ["train", "test", "val", "predict"]}
    data = make_multiple_dataframe_splits(root_dir / "tests" / "resources", transform_dict)
    subset = data["train"][range(len(data["train"]))]
    target_columns = ["seg1", "seg2"]
    batch_sampler = AlternatingBatchSampler(
        subset,
        target_columns=target_columns,
        batch_size=2,
        drop_last=True,
        shuffle=shuffle,
        sampler=sampler,
    )
    index_counts = np.zeros(len(subset))
    seg_type_counts = defaultdict(int)
    batches = []
    prev_type = None

    for batch in batch_sampler:
        na_cols = (
            ~subset.dataset.data.dataframe.take(batch).to_pandas()[target_columns].isna().any()
        )
        seg_type = na_cols.index[na_cols].tolist()

        assert len(seg_type) == 1
        seg_type_counts[seg_type[0]] += 1

        # alternate types when not shuffling
        if not shuffle and prev_type is not None:
            assert prev_type != seg_type
        prev_type = seg_type
        batches.append(batch)
    assert np.all(index_counts <= 1)  # no image should be sampled twice
    seg_counts = {v for k, v in seg_type_counts.items()}
    assert len(seg_counts) == 1  # all seg types are sampled equally
    assert len(seg_type_counts.keys()) == len(batch_sampler.samplers)  # all seg_types are sampled
    assert len(batches) == len(batch_sampler)  # yield same number of batches as len
