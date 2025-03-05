import numpy as np
from monai.data.meta_tensor import MetaTensor


def find_indices(lst, vals):
    arr = np.array(lst)

    sets = []
    for i, val in enumerate(vals):
        indices = set(np.where(arr == val)[0] - i)
        sets.append(indices)
    return np.asarray(list(set.intersection(*sets)), dtype=int)


def metatensor_batch_to_tensor(batch):
    """Convert monai metatensors to tensors."""
    for k, v in batch.items():
        if isinstance(v, MetaTensor):
            batch[k] = v.as_tensor()
    return batch
